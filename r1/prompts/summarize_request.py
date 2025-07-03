from server import mcp

@mcp.prompt()
def summarize_request(text: str) -> str:
    """Generate a prompt asking for a summary."""
    return f"Please summarize the following text:\n\n{text}"