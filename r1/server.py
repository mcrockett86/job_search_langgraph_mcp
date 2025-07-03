from mcp.server.fastmcp import FastMCP

# Initialize FastMCP server
mcp = FastMCP("r1-job-search", log_level="DEBUG")

# Import Tools
from tools.weather import *
from tools.arithmetic import *
from tools.scraper import *

# Import Prompts
from prompts.calculator import *
from prompts.summarize_request import *

# Import Resources
#from resources.greeting import *
#from resources.usage import *
from resources.version import *
from resources.resume import *


if __name__ == "__main__":
    import asyncio

    # use stdio for local testing the the mcp dev server
    asyncio.run(mcp.run(transport='stdio'))

    # for production, use streamable-http
    #asyncio.run(mcp.run(transport='streamable-http'))