# File: utils/helpers.py

import re

def clean_text(text: str) -> str:
    """
    Basic text cleaning: remove excessive whitespace, line breaks, etc.
    """
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces/newlines with single space
    return text.strip()
