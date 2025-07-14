from langchain_mcp_adapters.client import MultiServerMCPClient
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

from md2pdf.core import md2pdf
from linkedin_scraper import scrape_linkedin_job_urls

import pandas as pd
import asyncio
import os
import uuid


# load environment variables from .env file (e.g., OpenAI API key)
from dotenv import load_dotenv
load_dotenv('dotenv.env')

# set langchain verbose mode to True for debugging 
from langchain.globals import set_verbose
set_verbose(True)


def get_uuid_v4() -> str:
    return str(uuid.uuid4())


# Define the state for the job graph
class JobState(TypedDict):
    """State attributes and types of the job search graph."""

    id_run: str
    job_url: Dict[str, Any]
    job_description: Optional[str]
    is_match: Optional[bool]
    resume_draft: Optional[str]
    cover_letter_draft: Optional[str]
    messages: Annotated[list[AnyMessage], add_messages]


async def read_job_description(state: JobState, llm: ChatOpenAI) -> Dict:
    """Extract a detailed job description into markdown format from the provided job URL."""

    job_url = state["job_url"]

    # navigate to the job URL and extract the job description
    prompt = f"""
        You are a job seeker looking for a job that matches your skills and experience. 
        You have a URL for a specific job listing: {job_url}
        You have access to a tool 'scraper_extract_html_from_url' that can navigate to a URL and extract the webpage content.

        Use the tool 'scraper_extract_html' to navigate to the URL and extract the webpage content.

        The final answer is the webpage content from the provided URL.
        Return only the final answer.
    """

    messages = [HumanMessage(content=prompt)]
    response = await llm.ainvoke(messages)

    if len(state['messages']) > 0:
        last_message = state['messages'][-1]

        if type(last_message) == ToolMessage:
            if last_message.name == 'scraper_extract_html_from_url':
                job_description = last_message.content
                return {
                    "job_description": job_description,
                }
                
    return {
        "messages": [response]
    }


async def classify_job_description(state: JobState, llm: ChatOpenAI, resume_text: str):
    """Classify the job description to determine if it is a good fit for the job seeker's resume."""

    # only continue when the job_description is available in state
    if 'job_description' not in state:
        return {}
    else:
        job_description = state['job_description']
        
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

    return {
        "messages": [response]
    }

async def pending_node(state: JobState):
    print(f"Response Pending for job_url: {state['job_url']}")
    return {}

async def handle_bad_fit(state: JobState):
    print(f"Job seeker has marked this job description as a bad fit due to is_match: {state['is_match']}")
    return {}

async def handle_good_fit(state: JobState):
    print(f"Job seeker has marked this job description as a good fit due to is_match: {state['is_match']}")
    return {}

# Define routing logic
async def route_job(state: JobState) -> str:

    # only continue when the job_description is available in state
    if 'is_match' not in state:
        return "pending"

    else:
        if state["is_match"] == True:
            return "match"
        else:
            return "not_match"

async def draft_resume(state: JobState, llm: ChatOpenAI, resume_text: str):
    """Draft a Modification of Provided Resume for the Specific Job Description."""

    if 'resume_draft' not in state:
        print(f"Drafting a custom resume for the job seeker based on the job description and provided resume content.")
        job_description = state["job_description"]

        prompt = f"""
            You are an experienced job placement recruiter.  You are provided with a specific job description, and a job seeker's starting resume.  Make modifications to tailor the provided resume content with skills or interests that better match the provided job description.
            
            Job Description: {job_description}
            Resume Content: {resume_text}

            The final answer is the tailored resume.  Return only the final answer.\n\n\n\n
        """

        messages = [HumanMessage(content=prompt)]
        response = await llm.ainvoke(messages)

        if response.content:
            return {
                "resume_draft": response.content.lstrip("```markdown\n").rstrip("\n```"),
            }
        
        else:
            return {
                "messages": [response]
            }
    
    else:
        return state
    

async def draft_cover_letter(state: JobState, llm: ChatOpenAI, resume_text: str):
    """Draft a Modification of Provided Resume for the Specific Job Description."""

    if 'cover_letter_draft' not in state and 'resume_draft' in state:
        print(f"Drafting a custom cover letter for the job seeker based on the job description and provided resume content.")
        job_description = state["job_description"]
        resume_draft = state["resume_draft"]

        prompt = f"""
            You are a job seeker very interested in the job you are applying to.  
            You are provided with a specific job description, 
            You are provided with a resume that is tailored to that job description. 
            Create a cover letter tailored for applying to this job.
            
            Job Description: {job_description}
            Resume Content: {resume_draft}

            Do not include a detailed header with the recipient and sender information.  Instead, use this simplified format for the header: 'Dear Hiring Manager,'
            
            The final answer is the cover letter.  Return only the final answer.\n\n\n\n
        """

        messages = [HumanMessage(content=prompt)]
        response = await llm.ainvoke(messages)

        if response.content:
            return {
                "cover_letter_draft": response.content.lstrip("```markdown\n").rstrip("\n```"),
            }
        
        else:
            return {
                "messages": [response]
            }
    
    else:
        return state


async def save_resume_and_cover_letter(state: JobState):
    """Save the Drafted Resume for Job Seeker for Use Later."""

    id_run = state["id_run"]
    job_url = state["job_url"]
    resume_draft = state["resume_draft"]
    cover_letter_draft = state["cover_letter_draft"]

    # Save the draft resume to a file (converted to a PDF) and with a reference to the job URL
    print(f"Saving the draft resume and draft cover letter for the job seeker to apply to the job at {job_url}.")

    if not os.path.isdir('resumes'):
        os.mkdir('resumes')

    if not os.path.isdir('resumes_cover_letters'):
        os.mkdir('resumes_cover_letters')

    md2pdf(f"resumes/resume_{id_run}.pdf", md_content=resume_draft)
    md2pdf(f"resumes_cover_letters/cover_letter_{id_run}.pdf", md_content=cover_letter_draft)

    return {}


def create_job_graph(llm_with_tools: ChatOpenAI, tools: List[BaseTool], resources_dict: Dict[str,str], prompts: List[HumanMessage]) -> StateGraph:
    """
    Create a job graph for processing job descriptions and resumes.
    This function defines the nodes and edges of the job graph.
    """

    ### Instantiate the Graph Design from JobState and Create ToolNode from Tools List Provided ###
    builder = StateGraph(JobState)
    tool_node = ToolNode(tools)

    ### Create All Nodes in the Job Graph ###
    builder.add_node("tools", tool_node)
    builder.add_node("pending_node", pending_node)
    builder.add_node("read_job_description", partial(read_job_description, llm=llm_with_tools))
    builder.add_node("classify_job_description", partial(classify_job_description, llm=llm_with_tools, resume_text=resources_dict["resume_full"]))
    builder.add_node("handle_bad_fit", handle_bad_fit)
    builder.add_node("handle_good_fit", handle_good_fit)
    builder.add_node("draft_resume", partial(draft_resume, llm=llm_with_tools, resume_text=resources_dict["resume_full"]))
    builder.add_node("draft_cover_letter", partial(draft_cover_letter, llm=llm_with_tools, resume_text=resources_dict["resume_full"]))
    builder.add_node("save_resume_and_cover_letter", save_resume_and_cover_letter)

    ###  Define The Job Graph Routing Logic ###
    builder.add_edge(START, "read_job_description")
    builder.add_conditional_edges("read_job_description", tools_condition, ["tools", END])
    builder.add_edge("tools", "read_job_description")
    builder.add_edge("read_job_description", "classify_job_description")

    # Add conditional edges
    builder.add_conditional_edges(
        "classify_job_description",                # after classify, we run the "route_job" function"
        route_job,                                 # this function will return the next node to go to
        {
            "not_match": "handle_bad_fit",         # if the job description is a bad fit, return "not_match", go to the "handle_bad_fit" node
            "match":     "handle_good_fit",        # and if the job description is a good fit, return "match", go to the handle_good_fit" node
            "pending":   "pending_node"            # if classification not completed yet, stay at classify node
        }
    )
    builder.add_edge("handle_bad_fit", END)
    builder.add_edge("handle_good_fit", "draft_resume")
    builder.add_edge("draft_resume", "draft_cover_letter")


    # connect back up the pending node
    builder.add_edge("pending_node", "classify_job_description")

    # TODO: maybe need to make a create PDF tool from content?
    builder.add_edge("draft_cover_letter", "save_resume_and_cover_letter")

    builder.add_edge("save_resume_and_cover_letter", END)


    checkpointer = InMemorySaver()
    graph = builder.compile(checkpointer=checkpointer)

    mermaid_png_data = graph.get_graph().draw_mermaid_png()

    with open("job_graph.png", "wb") as f:
        f.write(mermaid_png_data)

    return graph


async def get_model_with_tools():

    client = MultiServerMCPClient(
        {
            #"r1_http": {
            #    "url": "http://localhost:8000/mcp",
            #    "transport": "streamable_http",
            #},

            # MCP Server Developed for Job Finder
            "r1_stdio": {
                "command": "python3",
                #"args": ["../r1/server.py"],  # relative or absolute path to mcp server script
                "args": ["r1/server.py"],  # relative or absolute path to mcp server script
                "transport": "stdio"
            },

            # MCP Server for Markdownify-MCP
            #"markdownify": {
            #    "command": "node",
            #    "args": [
            #        "markdownify-mcp/dist/index.js"
            #    ],
            #    "env": {
            #        #By default, the server will use the default install location of `uv`
            #        #"UV_PATH": "/path/to/uv"
            #    },
            #    "transport": "stdio"
            #}
        }
    )

    llm = init_chat_model(model="openai:gpt-4o-mini", temperature=0.0)

    # get the tools, prompts, and resources from the connected MCP servers for the MCP Client
    tools_r1_stdio = await client.get_tools(server_name="r1_stdio")         # get the tools from the r1_stdio-MCP server
    #tools_markdownify = await client.get_tools(server_name="markdownify")   # get the tools from the Markdownify-MCP server

    #tools = tools_r1_stdio + tools_markdownify                              # combine the set of tools from the 2 MCP servers
    tools = tools_r1_stdio

    resources_resume_full = await client.get_resources(server_name="r1_stdio", uris=["resume://full"])
    resources_resume_software = await client.get_resources(server_name="r1_stdio", uris=["resume://software"])
    resources_dict = {'resume_full': resources_resume_full[0].data, 'resume_software': resources_resume_software[0].data}

    prompts = await client.get_prompt(server_name="r1_stdio", prompt_name="summarize_request", arguments= {"text":"this is the text to summarize."})

    #print(f"tools:       {tools}\n\n")
    #print(f"resources:", resources_dict)
    #print(f"prompts:     {prompts}")

    llm_with_tools = llm.bind_tools(tools, parallel_tool_calls=False) # bind r1_stdio tools

    return llm, tools, llm_with_tools, resources_dict, prompts


async def main(df_targets: pd.DataFrame):

    ### Create the Job Graph Including the LLM with Tools from MCP Server ###
    llm, tools, llm_with_tools, resources_dict, prompts = await get_model_with_tools()
    job_graph = create_job_graph(llm_with_tools=llm_with_tools, tools=tools, resources_dict=resources_dict, prompts=prompts)

    for i, row in df_targets.iterrows():

        run_state = {
            "id_run": get_uuid_v4(),
            "job_url": row["job_url"],
            "messages": []
        }

        config = {"configurable": {"thread_id": i}}

        response = await job_graph.ainvoke(run_state, config, stream_mode='values')  # use stream_mode='values' to get streaming responses
        #response_state_dict = job_graph.get_state(config).values
        #job_graph_state = job_graph.get_state(config)

        try:
            df_targets.loc[i, 'thread_id'] = i
            df_targets.loc[i, 'id_run'] = response['id_run']
            df_targets.loc[i, 'is_match'] = response['is_match']
            #df_targets.loc[i, 'job_description'] = response['job_description']
            df_targets.loc[i, 'resume_path'] = os.path.join(os.path.abspath('resumes'), f"{response['id_run']}.pdf")

        except Exception as error:
            print(error)

    return df_targets


if __name__ == '__main__':

    # create the cache dir if not already present
    if not os.path.isdir('cache'):
        os.mkdir('cache')

    # first, call the linked_in_scraper to gather the available urls (sorted by relevance and sorted for past week jobs, mid sr level, location seattle, mid-sr-level)
    target_jobs_url = "https://www.linkedin.com/jobs/recommended/?f_E=4&f_PP=104116203&f_SB2=7&f_TPR=r604800&f_WT=1%2C3%2C2&origin=JOB_SEARCH_PAGE_JOB_FILTER&sortBy=R"
    df_new_jobs = scrape_linkedin_job_urls(target_jobs_url, max_pages=30)

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

        # if continue, update the cache
        df_old_jobs_update.to_csv('cache/target_jobs_old.csv', index=False)

        df_new_jobs_filtered['thread_id'] = ''
        df_new_jobs_filtered['id_run'] = ''
        df_new_jobs_filtered['is_match'] = ''
        df_new_jobs_filtered['resume_path'] = ''

        df_results = asyncio.run(main(df_new_jobs_filtered))
        df_results.to_csv('results_output.csv', index=False)

        # percent matches
        percent_matches = round(df_results['is_match'].sum() / len(df_new_jobs_filtered) * 100, 1)
        print(f"\n\nPercent matches:  {percent_matches}%")

    else:
        print(f"User selected to abort: {proceed}")
        quit()





