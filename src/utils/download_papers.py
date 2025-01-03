import arxiv
from pathlib import Path

class ArxivClient:
    """Class to interact with the arXiv API and handle paper downloads."""

    def __init__(self, max_results_per_topic=10, download_dir='./data/raw'):
        """
        Initialize the ArxivClient.
        
        :param max_results_per_topic: Maximum number of results to fetch per topic.
        :param download_dir: Directory to download PDF files.
        """
        self.max_results_per_topic = max_results_per_topic
        self.download_dir = Path(download_dir)  # Store the directory as a Path object
        self.download_dir.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists

    def download_papers(self, topics):
        """
        Download papers for the given topics from arXiv.
        
        :param topics: List of topics to search for.
        """
        for topic in topics:
            search = arxiv.Search(
                query=topic,
                max_results=self.max_results_per_topic,
                sort_by=arxiv.SortCriterion.SubmittedDate
            )

            for result in search.results():
                result.download_pdf(dirpath=self.download_dir)  # Save PDFs to the specified directory
