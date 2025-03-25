"""
ArXiv paper fetcher using the official arxiv library.
"""

import datetime
import logging
from typing import List, Dict, Optional
import arxiv
from time import sleep

logger = logging.getLogger(__name__)

class ArxivFetcher:
    """Fetches papers from ArXiv using the official API."""
    
    def __init__(self, max_results: int = 100, delay_seconds: float = 3.0):
        """
        Initialize the ArXiv fetcher.
        
        Args:
            max_results: Maximum number of results to return per query
            delay_seconds: Delay between API calls to respect rate limits
        """
        self.client = arxiv.Client()
        self.max_results = max_results
        self.delay_seconds = delay_seconds
        
    def _format_paper(self, paper: arxiv.Result) -> Dict:
        """Convert arxiv.Result to our standard paper format."""
        return {
            "id": paper.entry_id,
            "title": paper.title,
            "authors": [author.name for author in paper.authors],
            "abstract": paper.summary,
            "published": paper.published.strftime("%Y-%m-%d"),
            "updated": paper.updated.strftime("%Y-%m-%d"),
            "pdf_url": paper.pdf_url,
            "primary_category": paper.primary_category,
            "categories": paper.categories,
            "links": [link.href for link in paper.links],
            "comment": paper.comment
        }
    
    def get_papers(self, 
                   categories: List[str],
                   days_range: int = 7,
                   max_papers: Optional[int] = None) -> List[Dict]:
        """
        Fetch papers from specified categories within date range.
        
        Args:
            categories: List of arXiv categories (e.g., ['cs.AI', 'cs.LG'])
            days_range: Number of past days to look for papers
            max_papers: Maximum number of papers to return (None for no limit)
            
        Returns:
            List of papers in standardized format
        """
        max_results = max_papers if max_papers else self.max_results
        
        # Calculate date range
        end_date = datetime.datetime.now()
        start_date = end_date - datetime.timedelta(days=days_range)
        
        # Construct search query
        category_query = " OR ".join(f"cat:{cat}" for cat in categories)
        date_query = f"submittedDate:[{start_date.strftime('%Y%m%d')}* TO {end_date.strftime('%Y%m%d')}*]"
        search_query = f"({category_query}) AND {date_query}"
        
        logger.info(f"Searching arXiv with query: {search_query}")
        
        try:
            search = arxiv.Search(
                query=search_query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.SubmittedDate,
                sort_order=arxiv.SortOrder.Descending
            )
            
            results = []
            for paper in self.client.results(search):
                results.append(self._format_paper(paper))
                sleep(self.delay_seconds)  # Rate limiting
                
            logger.info(f"Successfully fetched {len(results)} papers")
            return results
            
        except Exception as e:
            logger.error(f"Error fetching papers from arXiv: {str(e)}")
            raise
            
    def get_paper_by_id(self, paper_id: str) -> Optional[Dict]:
        """
        Fetch a specific paper by its arXiv ID.
        
        Args:
            paper_id: ArXiv paper ID (can be with or without 'arXiv:' prefix)
            
        Returns:
            Paper data in standardized format or None if not found
        """
        try:
            # Remove 'arXiv:' prefix if present
            paper_id = paper_id.replace("arXiv:", "")
            
            search = arxiv.Search(
                id_list=[paper_id],
                max_results=1
            )
            
            paper = next(self.client.results(search), None)
            if paper:
                return self._format_paper(paper)
            return None
            
        except Exception as e:
            logger.error(f"Error fetching paper {paper_id}: {str(e)}")
            raise 