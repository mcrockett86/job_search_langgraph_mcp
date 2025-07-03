from server import mcp

# Static resource
@mcp.resource("config://version")
def get_version(): 
    return "r1"