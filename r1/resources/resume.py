from server import mcp
from pypdf import PdfReader


@mcp.resource("resume://software")
def get_resume_software() -> str:
    """Get resume including only software experience."""

    reader = PdfReader("docs/resume_software.pdf")

    text = ""
    for page in reader.pages:
        text += (page.extract_text() + "\n\n")

    return text

@mcp.resource("resume://full")
def get_resume_full() -> str:
    """Get resume including all experience, both software and mechanical engineering."""

    reader = PdfReader("docs/resume_full.pdf")

    text = ""
    for page in reader.pages:
        text += (page.extract_text() + "\n\n")

    return text