from server import mcp

@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers and return the result."""
    return a + b  # Simple arithmetic logic

@mcp.tool()
def subtract(a: int, b: int) -> int:
    """Subtract the second number from the first."""
    return a - b

@mcp.tool()
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b

@mcp.tool()
def divide(a: float, b: float) -> float:
    """Divide the first number by the second. Raises error on division by zero."""
    if b == 0:
        raise ValueError("Division by zero")
    return a / b