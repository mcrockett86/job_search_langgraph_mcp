from langchain.tools import tool
from pypdf import PdfReader
import os

async def get_resume_software() -> str:
    """Get resume including only software experience."""

    module_dir = os.path.dirname(__file__)
    fp = os.path.join(module_dir, 'resources', 'resume_software.pdf')
    
    reader = PdfReader(fp)

    text = ""
    for page in reader.pages:
        text += (page.extract_text() + "\n\n")

    return text

async def get_resume_full() -> str:
    """Get resume including all experience, both software and mechanical engineering."""

    module_dir = os.path.dirname(__file__)
    fp = os.path.join(module_dir, 'resources', 'resume_full.pdf')
    
    reader = PdfReader(fp)

    text = ""
    for page in reader.pages:
        text += (page.extract_text() + "\n\n")

    return text