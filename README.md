# job_search_langgraph_mcp
Job Search Agent Workflow Using LangGraph and MCP.  Scrapes available jobs from LinkedIn, gathers details of each job, and compares for match with content available from user-provided resume in PDF format as an MCP resource.  If good match, LangGraph Agentic pipeline generates a customized resume and cover letter for the specific job.  At the end of the workflow, a CSV file is presented containing a summary of the results, with links useful for human-in-the-loop application of the customized application content.  The software keeps a cache of prior jobs screened, so subsequent runs do not duplicate prior work.

## Install & Run
- install.sh
- run.sh

## Demo - Weekly Update Job Search



## LangGraph DAG
![Diagram Agent Flow](http://raw.githubusercontent.com/mcrockett86/job_search_langgraph_mcp/refs/heads/master/job_graph.png?raw=true)
