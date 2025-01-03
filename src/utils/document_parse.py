from llama_parse import LlamaParse
from pathlib import Path

class DocumentParser:
    """Class to parse the downloaded PDFs into markdown or text format."""

    def __init__(self, result_type="markdown", num_workers=4, verbose=True, api_key=None):
        """
        Initialize the DocumentParser.
        
        :param result_type: The format for the parsed result ('markdown' or 'text').
        :param num_workers: Number of parallel workers for parsing.
        :param verbose: Whether to enable verbose logging.
        :param api_key: The API key for LlamaParse (should be passed when initializing).
        """
        if api_key is None:
            raise ValueError("API key is required for LlamaParse.")
        
        self.parser = LlamaParse(
            result_type=result_type,
            num_workers=num_workers,
            verbose=verbose,
            api_key=api_key  # Pass the API key to LlamaParse
        )

    def parse_files(self, pdf_files, directory='./'):
        """
        Parse the list of PDF files in the specified directory.
        
        :param pdf_files: List of PDF filenames.
        :param directory: Directory where the PDF files are located.
        :return: List of parsed documents.
        """
        documents = []
        for index, pdf_file in enumerate(pdf_files):
            pdf_path = Path(directory) / pdf_file  # Build full path to the file
            print(f"Processing file {index + 1}/{len(pdf_files)}: {pdf_path}")
            document = self.parser.load_data(str(pdf_path))  # Pass the file path as a string
            documents.append(document)
        return documents
