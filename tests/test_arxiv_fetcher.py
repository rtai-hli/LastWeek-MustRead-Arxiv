"""Tests for the ArXiv fetcher module."""

import pytest
from datetime import datetime, timedelta
from src.fetchers.arxiv_fetcher import ArxivFetcher
from src.utils.sample_data import get_sample_papers

@pytest.fixture
def fetcher():
    """Create a fetcher instance with test settings."""
    return ArxivFetcher(max_results=5, delay_seconds=1.0)

@pytest.fixture
def sample_fetcher():
    """Create a fetcher instance that uses sample data."""
    return ArxivFetcher(max_results=5, delay_seconds=1.0, use_sample_data=True)

def test_fetcher_initialization():
    """Test fetcher initialization with custom parameters."""
    fetcher = ArxivFetcher(max_results=10, delay_seconds=2.0)
    assert fetcher.max_results == 10
    assert fetcher.delay_seconds == 2.0
    assert not fetcher.use_sample_data

def test_get_papers_basic(fetcher):
    """Test basic paper fetching functionality."""
    papers = fetcher.get_papers(categories=['cs.AI'], days_range=3, max_papers=2)
    
    assert isinstance(papers, list)
    assert len(papers) <= 2
    
    if papers:  # If any papers were returned
        paper = papers[0]
        required_fields = ['id', 'title', 'authors', 'abstract', 'published',
                         'updated', 'pdf_url', 'primary_category', 'categories']
        
        for field in required_fields:
            assert field in paper
            
        # Check date format
        datetime.strptime(paper['published'], '%Y-%m-%d')
        
        # Check if paper is within date range
        pub_date = datetime.strptime(paper['published'], '%Y-%m-%d')
        assert pub_date >= datetime.now() - timedelta(days=3)

def test_get_paper_by_id(fetcher):
    """Test fetching a specific paper by ID."""
    # Use a known paper ID for testing
    paper_id = "2401.00123"  # Replace with a known valid ID
    paper = fetcher.get_paper_by_id(paper_id)
    
    assert paper is not None
    assert paper['id'].endswith(paper_id)
    assert isinstance(paper['authors'], list)
    assert len(paper['authors']) > 0

def test_get_paper_by_id_with_prefix(fetcher):
    """Test fetching a paper using ID with arXiv prefix."""
    paper_id = "2401.00123"  # Replace with a known valid ID
    paper = fetcher.get_paper_by_id(f"arXiv:{paper_id}")
    
    assert paper is not None
    assert paper['id'].endswith(paper_id)

def test_get_paper_by_invalid_id(fetcher):
    """Test fetching a paper with invalid ID."""
    paper = fetcher.get_paper_by_id("invalid_id_123")
    assert paper is None

def test_get_papers_empty_result(fetcher):
    """Test fetching papers with criteria that should return no results."""
    # Use a very specific category and short time range
    papers = fetcher.get_papers(
        categories=['astro-ph.CO'],
        days_range=1,
        max_papers=1
    )
    assert isinstance(papers, list)  # Should return sample papers instead of empty list

def test_sample_data_fetching(sample_fetcher):
    """Test fetching papers using sample data."""
    papers = sample_fetcher.get_papers(categories=['cs.AI'], days_range=7)
    sample_papers = get_sample_papers()
    
    assert len(papers) == len(sample_papers)
    assert papers[0]['id'] == sample_papers[0]['id']
    assert papers[0]['title'] == sample_papers[0]['title']

def test_sample_data_paper_by_id(sample_fetcher):
    """Test fetching a specific paper from sample data."""
    paper = sample_fetcher.get_paper_by_id("sample1")
    assert paper is not None
    assert paper['id'] == "sample1"
    
    # Test with invalid ID
    paper = sample_fetcher.get_paper_by_id("nonexistent")
    assert paper is None

def test_sample_data_fields(sample_fetcher):
    """Test that sample data papers have all required fields."""
    papers = sample_fetcher.get_papers(categories=['cs.AI'])
    required_fields = ['id', 'title', 'authors', 'abstract', 'published',
                      'updated', 'pdf_url', 'primary_category', 'categories',
                      'links', 'comment']
    
    for paper in papers:
        for field in required_fields:
            assert field in paper 