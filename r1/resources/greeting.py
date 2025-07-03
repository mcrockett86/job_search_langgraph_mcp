from server import mcp

# Add a dynamic greeting resource
@mcp.resource("greeting://greet/{name}")
def job_search_greeting(name: str) -> str:
    """Get a personalized greeting"""
    return f"Hello, {name}! Ready to find a job today?"