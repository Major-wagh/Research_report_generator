from pathlib import Path

def list_pdf_files(directory='./'):
    """
    List all .pdf files in the specified directory.
    
    :param directory: Directory to search for PDF files.
    :return: List of PDF file names.
    """
    pdf_files = [file.name for file in Path(directory).glob('*.pdf')]
    return pdf_files
