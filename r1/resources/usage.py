from server import mcp
import os


@mcp.resource("usage://guide")
def get_usage() -> str:
    with open("docs/usage.txt") as f:
        return f.read()