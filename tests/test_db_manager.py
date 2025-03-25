"""Tests for the database manager module."""

import pytest
import os
import sqlite3
from datetime import datetime, timedelta
from src.database.db_manager import DatabaseManager
from src.utils.sample_data import get_sample_papers

@pytest.fixture
def test_db_path(tmp_path):
    """Create a temporary database path."""
    return str(tmp_path / "test.db")

@pytest.fixture
def db_manager(test_db_path):
    """Create a database manager instance with test database."""
    manager = DatabaseManager(test_db_path)
    manager.initialize_database()
    return manager

@pytest.fixture
def sample_paper():
    """Get a sample paper for testing."""
    return {
        "paper_id": "test123",
        "title": "Test Paper",
        "authors": ["John Doe", "Jane Smith"],
        "published_date": "2024-03-25",
        "processed_date": "2024-03-25",
        "summary": "This is a test paper summary",
        "classification": {
            "category": "Test Category",
            "confidence": 0.9,
            "rationale": "Test rationale"
        },
        "novelty_assessment": {
            "score": 8,
            "level": "High",
            "description": "Test description",
            "strengths": ["strength1"],
            "limitations": ["limitation1"]
        },
        "score": 8.5,
        "scoring_rationale": "Test scoring rationale"
    }

def test_database_initialization(test_db_path):
    """Test database initialization."""
    manager = DatabaseManager(test_db_path)
    manager.initialize_database()
    
    # Check if database file exists
    assert os.path.exists(test_db_path)
    
    # Check if table exists with correct schema
    conn = sqlite3.connect(test_db_path)
    cursor = conn.cursor()
    
    # Get table info
    cursor.execute("PRAGMA table_info(papers)")
    columns = {row[1]: row[2] for row in cursor.fetchall()}
    
    # Check required columns
    assert "id" in columns
    assert "title" in columns
    assert "authors" in columns
    assert "score" in columns
    assert columns["score"] == "REAL"
    
    conn.close()

def test_save_paper(db_manager, sample_paper):
    """Test saving a paper to the database."""
    success = db_manager.save_paper_analysis(sample_paper)
    assert success
    
    # Verify paper was saved
    saved_paper = db_manager.get_paper_by_id(sample_paper["paper_id"])
    assert saved_paper is not None
    assert saved_paper["title"] == sample_paper["title"]
    assert saved_paper["score"] == sample_paper["score"]
    assert isinstance(saved_paper["authors"], list)

def test_save_invalid_paper(db_manager):
    """Test saving a paper with missing fields."""
    invalid_paper = {
        "paper_id": "invalid123",
        "title": "Invalid Paper"
        # Missing required fields
    }
    
    with pytest.raises(ValueError) as exc_info:
        db_manager.save_paper_analysis(invalid_paper)
    assert "Missing required fields" in str(exc_info.value)

def test_get_papers_by_date(db_manager):
    """Test retrieving papers by date."""
    # Save sample papers with different dates
    papers = get_sample_papers()
    today = datetime.now().strftime("%Y-%m-%d")
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    
    for i, paper in enumerate(papers):
        paper_data = paper.copy()
        paper_data["paper_id"] = paper["id"]
        paper_data["processed_date"] = today if i % 2 == 0 else yesterday
        db_manager.save_paper_analysis(paper_data)
    
    # Test retrieval
    today_papers = db_manager.get_papers_by_date(today)
    assert len(today_papers) == len(papers) // 2
    
    yesterday_papers = db_manager.get_papers_by_date(yesterday)
    assert len(yesterday_papers) == len(papers) // 2

def test_get_top_papers(db_manager):
    """Test retrieving top papers by score."""
    # Save sample papers
    papers = get_sample_papers()
    for i, paper in enumerate(papers):
        paper_data = paper.copy()
        paper_data["paper_id"] = paper["id"]
        paper_data["score"] = 10 - i  # Descending scores
        db_manager.save_paper_analysis(paper_data)
    
    # Test retrieval
    top_2_papers = db_manager.get_top_papers(n=2)
    assert len(top_2_papers) == 2
    assert top_2_papers[0]["score"] > top_2_papers[1]["score"]

def test_get_statistics(db_manager):
    """Test retrieving database statistics."""
    # Save sample papers
    papers = get_sample_papers()
    for paper in papers:
        paper_data = paper.copy()
        paper_data["paper_id"] = paper["id"]
        db_manager.save_paper_analysis(paper_data)
    
    stats = db_manager.get_statistics()
    assert stats["total_papers"] == len(papers)
    assert stats["avg_score"] > 0
    assert stats["max_score"] > 0
    assert stats["first_date"] is not None
    assert stats["last_date"] is not None
    assert len(stats["top_categories"]) > 0

def test_paper_not_found(db_manager):
    """Test retrieving a non-existent paper."""
    paper = db_manager.get_paper_by_id("nonexistent")
    assert paper is None

def test_get_top_papers_with_timeframe(db_manager):
    """Test retrieving top papers within a time window."""
    # Save sample papers with different dates
    papers = get_sample_papers()
    today = datetime.now().strftime("%Y-%m-%d")
    old_date = (datetime.now() - timedelta(days=10)).strftime("%Y-%m-%d")
    
    for i, paper in enumerate(papers):
        paper_data = paper.copy()
        paper_data["paper_id"] = paper["id"]
        paper_data["processed_date"] = today if i % 2 == 0 else old_date
        paper_data["score"] = 10 - i
        db_manager.save_paper_analysis(paper_data)
    
    # Get top papers from last 7 days
    recent_top_papers = db_manager.get_top_papers(n=10, days=7)
    assert all(p["processed_date"] == today for p in recent_top_papers)
    assert len(recent_top_papers) == len(papers) // 2 