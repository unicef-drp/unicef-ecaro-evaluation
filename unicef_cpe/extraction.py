"""
Functions for extracting text from files and data from web APIs.
"""

import os
from io import BytesIO

import pandas as pd
import requests
from docx import Document
from pypdfium2 import PdfDocument
from langchain_community.document_loaders import  PyPDFium2Loader
from langchain.document_loaders import PyPDFium2Loader

def extract_text_using_ocr(file_path: str) -> str:
    try:
        # Importing pdf2image and pytesseract inside the function
        from pdf2image import convert_from_path
        import pytesseract
    except ImportError:
        print("OCR libraries (pdf2image and pytesseract) are not installed. OCR functionality is unavailable.")
        return ""

    """
    Extract text from a PDF using OCR (Tesseract) after converting it to images.

    Parameters
    ----------
    file_path : str
        Path to a PDF file.

    Returns
    -------
    ocr_text : str
        Full text extracted from the file using OCR.
    """
    ocr_texts = []
    try:
        # Convert PDF pages to images
        images = convert_from_path(file_path)
        for i, image in enumerate(images):
            ocr_text = pytesseract.image_to_string(image)  # Perform OCR
            ocr_texts.append(ocr_text)

        # Join all the OCR results
        return " ".join(ocr_texts)

    except Exception as ocr_error:
        print(f"Error during OCR process: {ocr_error}")
        return ""  # Return empty string if OCR fails


def extract_text_from_pdf(file_path: str) -> str:
    """
    Extract text from a PDF using PDFium bindings or fallback to OCR if needed.

    Parameters
    ----------
    file_path : str
        Path to a PDF file.

    Returns
    -------
    text : str
        Full text extracted from the file.
    """
    # Step 1: Try extracting text with PyPDFium2Loader
    loader = PyPDFium2Loader(file_path)
    pdf = loader.load()

    texts = []
    for page in pdf:
        try:
            text = page.page_content  # Extract text content
        except Exception as e:
            print(f"Error extracting text from page: {e}")
            text = ""
        finally:
            texts.append(text)

    # Join the extracted text
    extracted_text = " ".join(texts)

    # Step 2: Check if the extracted text is empty and fallback to OCR if necessary
    if extracted_text.strip() == "":
        print("No text found using standard extraction. Falling back to OCR.")
        extracted_text = extract_text_using_ocr(file_path)

    return extracted_text


def extract_text_from_docx(file_path: str) -> str:
    """
    Extract text from a Word document (.docx only).

    Parameters
    ----------
    file_path : str
        Path to a Word file. Only .docx files are supported.

    Returns
    -------
    text : str
        Full text extracted from the file.
    """
    document = Document(file_path)
    text = "\n".join(paragraph.text for paragraph in document.paragraphs)
    return text


def extract_text_from_file(file_path: str) -> str:
    """
    Extract text from a PDF (.pdf) or Word (.docx) document.

    Parameters
    ----------
    file_path : str
        Path to a PDF or Word document.

    Returns
    -------
    text : str
        Full text extracted from the file.
    """
    _, extension = os.path.splitext(file_path)
    if extension == ".pdf":
        text = extract_text_from_pdf(file_path)
    elif extension == ".docx":
        text = extract_text_from_docx(file_path)
    else:
        raise ValueError(
            f"{extension} files are not supported. Only .docx or .pdf files can be processed."
        )
    return text


def extract_from_sdmx(
    dataflow: str = "UNICEF,GLOBAL_DATAFLOW,1.0", pad: int = 0, **kwargs
) -> pd.DataFrame:
    """
    Get from UNICEF Indicator Data Warehouse (SDMX).
    """
    url_base = "https://sdmx.data.unicef.org/ws/public/sdmxapi/rest/data"
    filters = ""
    if kwargs:
        for values in kwargs.values():
            assert isinstance(
                values, list
            ), "Keyword arguments must be lists of strings."
            filters = "+".join(values) + "."
    padding = "." * pad
    endpoint = f"{url_base}/{dataflow}/{filters}{padding}"
    params = {
        "format": "csv",
    }
    response = requests.get(endpoint, params=params)
    response.raise_for_status()
    df = pd.read_csv(BytesIO(response.content), low_memory=False)
    return df
