from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, AnyMessage
from langchain_core.tools import BaseTool, InjectedToolArg, StructuredTool, ToolException

from langchain_openai import ChatOpenAI
from langchain.chat_models import init_chat_model

from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent, tools_condition, ToolNode

from typing import TypedDict, List, Dict, Any, Optional, Annotated
from functools import partial

from linkedin_scraper import scrape_linkedin_job_urls
from tools.resume import get_resume_software, get_resume_full
from tools.scraper import scraper_extract_html_from_url


import pandas as pd
import asyncio
import os
import uuid
import json
import pdb

# load environment variables from .env file (e.g., OpenAI API key)
from dotenv import load_dotenv
load_dotenv('dotenv.env')

def get_uuid_v4() -> str:
    return str(uuid.uuid4())


# Define the state for the job graph
class JobState(TypedDict):
    """State attributes and types of the job search graph."""

    id_run: str
    job_url: Dict[str, Any]
    job_description: Optional[str]
    is_match: Optional[bool]
    company_name: Optional[str]
    job_title: Optional[str]    
    job_type: Optional[str]
    salary_min: Optional[float]
    salary_max: Optional[float]   
    messages: Annotated[list[AnyMessage], add_messages]


async def read_job_description(state: JobState) -> Dict:
    """Extract a detailed job description into markdown format from the provided job URL."""

    job_url = state["job_url"]

    # call tool to navigate to the job URL and extract the job description
    html_content_text = await scraper_extract_html_from_url(job_url)

    return {
        "job_description": html_content_text
    }
    

async def classify_job_description(state: JobState):
    """Classify the job description to determine if it is a good fit for the job seeker's resume."""

    llm = init_chat_model(model="openai:gpt-4o-mini", temperature=0.0)

    job_description = state['job_description']
    resume_text = await get_resume_software()
   
    prompt = f"""
        You are a job seeker looking for a job that matches your skills and experience.
        You have a detailed job description in text format that contains the required skills and experience for the role.
        You have a detailed resume in text format that contains your skills and experience.
                
        Analyze the job description and compare it to the resume content, and determine if this job is a good fit for the prospective job seeker.
        The final answer should be either "yes" or "no".  Return only the final answer.\n\n\n\n

        Job Description: {job_description}\n\n\n\n
        Resume Content:  {resume_text}\n\n\n\n
    """

    messages = [HumanMessage(content=prompt)]
    response = await llm.ainvoke(messages)

    if response.content:
        response_text = response.content.lower()
        is_match = "yes" in response_text or "good" in response_text

        return {
            "is_match": is_match
        }

    else:
        return {
            "is_match": False
        }


async def extract_job_details(state: JobState):
    """Extract Job Details from the Job Description."""

    llm = init_chat_model(model="openai:gpt-4o-mini", temperature=0.0)

    job_description = state['job_description']
    resume_text = await get_resume_software()
   
    prompt = f'''
        You are an information extraction agent operating inside a LangGraph workflow.

        Your task is to extract structured job metadata from a raw job listing unstructured text provided below as INPUT:

        INPUT:
        {job_description}

        OUTPUT REQUIREMENTS:
        - Return a JSON-formatted string ONLY.
        - Do not include markdown, explanations, or extra text.
        - Use null for any field that cannot be confidently determined.
        - If salary is expressed as a range, extract both min and max.
        - If salary is expressed as a single value, set both salary_min and salary_max to that value.
        - If salary is expressed in words (e.g., "six figures"), estimate conservatively and document internally, but still output numeric values.
        - All salary values must be numeric and represent annual compensation in USD.
        - Do not infer or hallucinate values that are not present or reasonably implied.

        FIELDS TO EXTRACT:
        - company_name: string | null (examples: "Meta", "Amazon", "Perplexity")
        - job_title: string | null (examples: "Data Scientist", "Machine Learning Engineer", "AI Engineer")
        - job_type: string | null (examples: "Full-time", "Part-time", "Contract")
        - salary_min: number | null (examples: 100000, 150000)
        - salary_max: number | null (examples: 150000, 200000)

        NORMALIZATION RULES:
        - Trim whitespace from all string fields.
        - Preserve original capitalization for company_name and job_title.
        - job_type should be normalized to a concise canonical form if possible.

        Begin extraction now.

    '''

    messages = [HumanMessage(content=prompt)]
    response = await llm.ainvoke(messages)

    try:
        if response.content:
            response_text = response.content.lower()

            # parse the text and convert to dict
            json_dict = json.loads(response_text)

            # verify that keys are present, and if not set them to none before updating the state
            for k in ["company_name", "job_title", "job_type", "salary_min", "salary_max"]:
                if k not in json_dict:
                    json_dict[k] = None

            return json_dict

    except Exception as error:
        print(error)

        return {
            "company_name": None,
            "job_title": None,
            "job_type": None,
            "salary_min": None,
            "salary_max": None
        }


def handle_bad_fit(state: JobState):
    print(f"Job seeker has marked this job description as a bad fit due to is_match: {state['is_match']}")
    return {}

def handle_good_fit(state: JobState):
    print(f"Job seeker has marked this job description as a good fit due to is_match: {state['is_match']}")
    return {}


# Define routing logic
async def route_job(state: JobState) -> str:

    # only continue when the job_description is available in state
    if state["is_match"] == True:
        return "match"
    else:
        return "not_match"


def create_job_graph() -> StateGraph:
    """
    Create a job graph for processing job descriptions and resumes.
    This function defines the nodes and edges of the job graph.
    """

    ### Instantiate the Graph Design from JobState and Create ToolNode from Tools List Provided ###
    builder = StateGraph(JobState)

    ### Create All Nodes in the Job Graph ###
    builder.add_node("read_job_description", read_job_description)
    builder.add_node("classify_job_description", classify_job_description)
    builder.add_node("extract_job_details", extract_job_details)
    builder.add_node("handle_bad_fit", handle_bad_fit)
    builder.add_node("handle_good_fit", handle_good_fit)

    ###  Define The Job Graph Routing Logic ###
    builder.add_edge(START, "read_job_description")
    builder.add_edge("read_job_description", "classify_job_description")

    # Add conditional edges
    builder.add_conditional_edges(
        "classify_job_description",                # after classify, we run the "route_job" function"
        route_job,                                 # this function will return the next node to go to
        {
            "not_match": "handle_bad_fit",         # if the job description is a bad fit, return "not_match", go to the "handle_bad_fit" node
            "match":     "handle_good_fit"        # and if the job description is a good fit, return "match", go to the handle_good_fit" node
        }
    )
    builder.add_edge("handle_bad_fit", END)
    builder.add_edge("handle_good_fit", "extract_job_details")
    builder.add_edge("extract_job_details", END)

    checkpointer = InMemorySaver()
    graph = builder.compile(checkpointer=checkpointer)

    mermaid_png_data = graph.get_graph().draw_mermaid_png()

    with open("job_graph.png", "wb") as f:
        f.write(mermaid_png_data)

    return graph


async def main(df_targets: pd.DataFrame):

    ### Create the Job Graph Including the LLM with Tools from MCP Server ###
    job_graph = create_job_graph()

    for i, row in df_targets.iterrows():
        try:

            run_state = {
                "id_run": get_uuid_v4(),
                "job_url": row["job_url"],
                "messages": []
            }

            config = {"configurable": {"thread_id": i}}
            response = await job_graph.ainvoke(run_state, config, stream_mode='values')

            df_targets.loc[i, 'thread_id'] = i
            # df_targets.loc[i, 'id_run'] = response['id_run']
            df_targets.loc[i, 'is_match'] = response['is_match']
            df_targets.loc[i, 'company_name'] = response['company_name']
            df_targets.loc[i, 'job_title'] = response['job_title']
            df_targets.loc[i, 'job_type'] = response['job_type']
            df_targets.loc[i, 'salary_min'] = response['salary_min']
            df_targets.loc[i, 'salary_max'] = response['salary_max']

        except Exception as error:
            print(error)

    return df_targets


if __name__ == '__main__':

    # create the cache dir if not already present
    if not os.path.isdir('cache'):
        os.mkdir('cache')

    # first, call the linked_in_scraper to gather the available urls (sorted by relevance and sorted for past week jobs, mid sr level, location seattle, bellevue, kirkland, entry level or mid-sr-level)    
    target_jobs_urls = [

        # Only Remote (weekly)
        "https://www.linkedin.com/jobs/search/?f_E=2%2C4&f_SB2=6&f_TPR=r604800&f_WT=2&geoId=103644278&keywords=data%20scientist&origin=JOB_SEARCH_PAGE_JOB_FILTER&refresh=true&sortBy=R",

        "https://www.linkedin.com/jobs/search/?f_E=2%2C4&f_SB2=6&f_TPR=r604800&f_WT=2&geoId=103644278&keywords=artificial%20intelligence&origin=JOB_SEARCH_PAGE_JOB_FILTER&refresh=true&sortBy=R",

        "https://www.linkedin.com/jobs/search/?f_E=2%2C4&f_SB2=6&f_TPR=r604800&f_WT=2&geoId=103644278&keywords=machine%20learning&origin=JOB_SEARCH_PAGE_JOB_FILTER&refresh=true&sortBy=R",


        # Seattle, WA (weekly)
        "https://www.linkedin.com/jobs/search/?f_E=2%2C4&f_PP=104116203%2C106619589%2C104145663&f_SB2=6&f_TPR=r604800&f_WT=1%2C3%2C2&geoId=103644278&keywords=data%20scientist&origin=JOB_SEARCH_PAGE_JOB_FILTER&refresh=true&sortBy=R",

        "https://www.linkedin.com/jobs/search/?f_E=2%2C4&f_PP=104116203%2C106619589%2C104145663&f_SB2=6&f_TPR=r604800&f_WT=1%2C3%2C2&geoId=103644278&keywords=artificial%20intelligence&origin=JOB_SEARCH_PAGE_JOB_FILTER&refresh=true&sortBy=R",

        "https://www.linkedin.com/jobs/search/?f_E=2%2C4&f_PP=104116203%2C106619589%2C104145663&f_SB2=6&f_TPR=r604800&f_WT=1%2C3%2C2&geoId=103644278&keywords=machine%20learning&origin=JOB_SEARCH_PAGE_JOB_FILTER&refresh=true&sortBy=R",

        "https://www.linkedin.com/jobs/collections/recommended/?discover=recommended&discoveryOrigin=JOBS_HOME_JYMBII",


        # San Diego, CA (weekly)
        "https://www.linkedin.com/jobs/search/?f_E=2%2C4&f_PP=103918656&f_SB2=6&f_TPR=r604800&f_WT=1%2C3%2C2&geoId=90010472&keywords=data%20scientist&refresh=true&sortBy=R",

        "https://www.linkedin.com/jobs/search/?f_E=2%2C4&f_PP=103918656&f_SB2=6&f_TPR=r604800&f_WT=2%2C1%2C3&geoId=90010472&keywords=artificial%20intelligence&origin=JOB_SEARCH_PAGE_JOB_FILTER&refresh=true&sortBy=R",

        "https://www.linkedin.com/jobs/search/?f_E=2%2C4&f_PP=103918656&f_SB2=6&f_TPR=r604800&f_WT=2%2C1%2C3&geoId=90010472&keywords=machine%20learning&origin=JOB_SEARCH_PAGE_JOB_FILTER&refresh=true&sortBy=R",


        # San Francisco Bay Area, CA (weekly)
        "https://www.linkedin.com/jobs/search/?f_E=2%2C4&f_PP=102277331&f_SB2=6&f_TPR=r604800&f_WT=1%2C3%2C2&geoId=90000084&keywords=data%20scientist&origin=JOB_SEARCH_PAGE_JOB_FILTER&refresh=true&sortBy=R",

        "https://www.linkedin.com/jobs/search/?f_E=2%2C4&f_PP=102277331&f_SB2=6&f_TPR=r604800&f_WT=1%2C3%2C2&geoId=90000084&keywords=artificial%20intelligence&origin=JOB_SEARCH_PAGE_JOB_FILTER&refresh=true&sortBy=R",

        "https://www.linkedin.com/jobs/search/?f_E=2%2C4&f_PP=102277331&f_SB2=6&f_TPR=r604800&f_WT=1%2C3%2C2&geoId=90000084&keywords=machine%20learning&origin=JOB_SEARCH_PAGE_JOB_FILTER&refresh=true&sortBy=R",


        # Los Angeles, CA (weekly)
        "https://www.linkedin.com/jobs/search/?f_E=2%2C4&f_PP=102448103&f_SB2=6&f_TPR=r604800&f_WT=1%2C2%2C3&geoId=90000049&keywords=data%20scientist&origin=JOB_SEARCH_PAGE_JOB_FILTER&refresh=true&sortBy=R",

        "https://www.linkedin.com/jobs/search/?f_E=2%2C4&f_PP=102448103&f_SB2=6&f_TPR=r604800&f_WT=1%2C2%2C3&geoId=90000049&keywords=artificial%20intelligence&origin=JOB_SEARCH_PAGE_JOB_FILTER&refresh=true&sortBy=R",

        "https://www.linkedin.com/jobs/search/?f_E=2%2C4&f_PP=102448103&f_SB2=6&f_TPR=r604800&f_WT=1%2C2%2C3&geoId=90000049&keywords=machine%20learning&origin=JOB_SEARCH_PAGE_JOB_FILTER&refresh=true&sortBy=R"
    ]
        
    df_new_jobs = scrape_linkedin_job_urls(target_jobs_urls, max_pages=10)

    # drop duplicate links found (overlapping)
    df_new_jobs = df_new_jobs.drop_duplicates()

    # load df for prior jobs processed
    df_prior_jobs = pd.read_csv("cache/target_jobs_old.csv")

    # compare the new ones to prior applications, and remove prior processed applications from the list
    df_new_jobs_filtered = df_new_jobs[~df_new_jobs['job_url'].isin(df_prior_jobs['job_url'])].reset_index()
    df_old_jobs_update = pd.concat([df_new_jobs_filtered, df_prior_jobs], ignore_index=True)

    print(f"new jobs detected: {len(df_new_jobs_filtered.index)}")
    print(df_new_jobs_filtered)

    # prompt user to continue if wanted
    proceed = input("\n\nContinue job search agentic pipeline workflow? Type 'yes' to continue: ")

    if proceed == 'yes':
        print("Updating prior jobs application cache ...")

        df_old_jobs_update.to_csv('cache/target_jobs_old.csv', index=False)

        df_new_jobs_filtered['thread_id'] = ''
        df_new_jobs_filtered['id_run'] = ''
        df_new_jobs_filtered['is_match'] = ''

        df_new_jobs_filtered['company_name'] = ''
        df_new_jobs_filtered['job_title'] = ''
        df_new_jobs_filtered['job_type'] = ''
        df_new_jobs_filtered['salary_min'] = ''
        df_new_jobs_filtered['salary_max'] = ''

        df_results = asyncio.run(main(df_new_jobs_filtered))
        
        # percent matches
        percent_matches = round(df_results['is_match'].sum() / len(df_new_jobs_filtered) * 100, 1)
        print(f"\n\nPercent matches:  {percent_matches}%")

        # export only the matches
        df_results = df_results[df_results['is_match'] == True]

        # TODO: remove any company names not desired
        remove_companies = ['amazon', 'harnham', 'insight global', 'tiktok', 'walmart', 'microsoft', 'apple']

        df_results_filtered = df_results[~df_results['company_name'].isin(remove_companies)]

        df_results_filtered.to_csv('results_output.csv', index=False)

    else:
        print(f"User selected to abort: {proceed}")
        quit()