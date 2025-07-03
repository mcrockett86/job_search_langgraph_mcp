from server import mcp
from tools.arithmetic import add, subtract, multiply, divide


@mcp.prompt()
def calculator_prompt(a: float, b: float, operation: str) -> str:
    """Prompt for a calculation and return the result."""
    if operation == "add":
        return f"The result of adding {a} and {b} is {add(a, b)}"
    elif operation == "subtract":
        return f"The result of subtracting {b} from {a} is {subtract(a, b)}"
    elif operation == "multiply":
        return f"The result of multiplying {a} and {b} is {multiply(a, b)}"
    elif operation == "divide":
        try:
            return f"The result of dividing {a} by {b} is {divide(a, b)}"
        except ValueError as e:
            return str(e)
    else:
        return "Invalid operation. Please choose add, subtract, multiply, or divide."