from server import mcp

@mcp.resource("jobs://available")
def get_jobs_available(url: str) -> str:
    """Get available jobs listed from a specific company jobs career page."""

    jobs_listed = PdfReader("docs/resume_software.pdf")

    text = ""
    for page in reader.pages:
        text += (page.extract_text() + "\n\n")

    return text