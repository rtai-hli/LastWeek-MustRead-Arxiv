# database/db_manager.py
import logging
import sqlite3
import json
from typing import Dict, Any, List, Optional
import pandas as pd
import os
from datetime import datetime

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Manages the database for paper analysis results."""
    
    def __init__(self, db_path: str):
        """
        Initialize the database manager.
        
        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = db_path
        
    def initialize_database(self) -> None:
        """Initialize the database structure."""
        try:
            # Ensure database directory exists
            os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
            
            # Create database connection
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create table structure
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS papers (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                authors TEXT NOT NULL,
                published_date TEXT NOT NULL,
                processed_date TEXT NOT NULL,
                summary TEXT NOT NULL,
                classification TEXT NOT NULL,
                novelty_assessment TEXT NOT NULL,
                score REAL NOT NULL,
                scoring_rationale TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            ''')
            
            # Create indices for common queries
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_processed_date ON papers(processed_date)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_score ON papers(score)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_published_date ON papers(published_date)')
            
            conn.commit()
            logger.info(f"Successfully initialized database: {self.db_path}")
            
        except Exception as e:
            logger.error(f"Error initializing database: {str(e)}")
            raise
            
        finally:
            if 'conn' in locals():
                conn.close()
    
    def save_paper_analysis(self, paper_result: Dict[str, Any]) -> bool:
        """
        Save paper analysis results to the database.
        
        Args:
            paper_result: Dictionary containing paper analysis results
            
        Returns:
            Whether the save was successful
            
        Raises:
            ValueError: If required fields are missing
        """
        required_fields = [
            "paper_id", "title", "authors", "published_date", "processed_date",
            "summary", "classification", "novelty_assessment", "score", "scoring_rationale"
        ]
        
        # Validate required fields
        missing_fields = [field for field in required_fields if field not in paper_result]
        if missing_fields:
            raise ValueError(f"Missing required fields: {', '.join(missing_fields)}")
            
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Prepare data
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Convert authors list to string if necessary
            authors = paper_result["authors"]
            if isinstance(authors, list):
                authors = ", ".join(authors)
            
            # Insert or update data
            cursor.execute('''
            INSERT OR REPLACE INTO papers (
                id, title, authors, published_date, processed_date,
                summary, classification, novelty_assessment, score, scoring_rationale,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                paper_result["paper_id"],
                paper_result["title"],
                authors,
                paper_result["published_date"],
                paper_result["processed_date"],
                paper_result["summary"],
                json.dumps(paper_result["classification"], ensure_ascii=False),
                json.dumps(paper_result["novelty_assessment"], ensure_ascii=False),
                paper_result["score"],
                paper_result["scoring_rationale"],
                now,
                now
            ))
            
            conn.commit()
            logger.info(f"Successfully saved paper analysis: {paper_result['title']}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving paper analysis: {str(e)}")
            return False
            
        finally:
            if 'conn' in locals():
                conn.close()
    
    def get_papers_by_date(self, date_str: str) -> List[Dict[str, Any]]:
        """
        Get papers processed on a specific date.
        
        Args:
            date_str: Date string in YYYY-MM-DD format
            
        Returns:
            List of paper dictionaries
        """
        try:
            conn = sqlite3.connect(self.db_path)
            
            query = "SELECT * FROM papers WHERE DATE(processed_date) = DATE(?)"
            papers_df = pd.read_sql_query(query, conn, params=(date_str,))
            
            # Process JSON fields
            papers_df["classification"] = papers_df["classification"].apply(json.loads)
            papers_df["novelty_assessment"] = papers_df["novelty_assessment"].apply(json.loads)
            papers_df["authors"] = papers_df["authors"].apply(lambda x: x.split(", "))
            
            papers = papers_df.to_dict(orient="records")
            logger.info(f"Retrieved {len(papers)} papers from {date_str}")
            return papers
            
        except Exception as e:
            logger.error(f"Error retrieving papers: {str(e)}")
            return []
            
        finally:
            if 'conn' in locals():
                conn.close()
    
    def get_top_papers(self, n: int = 10, days: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get top N papers by score, optionally within a recent time window.
        
        Args:
            n: Number of papers to return
            days: If provided, only consider papers from the last N days
            
        Returns:
            List of paper dictionaries
        """
        try:
            conn = sqlite3.connect(self.db_path)
            
            if days:
                query = """
                SELECT * FROM papers 
                WHERE DATE(processed_date) >= DATE('now', ?)
                ORDER BY score DESC LIMIT ?
                """
                params = (f'-{days} days', n)
            else:
                query = "SELECT * FROM papers ORDER BY score DESC LIMIT ?"
                params = (n,)
                
            papers_df = pd.read_sql_query(query, conn, params=params)
            
            # Process JSON fields
            papers_df["classification"] = papers_df["classification"].apply(json.loads)
            papers_df["novelty_assessment"] = papers_df["novelty_assessment"].apply(json.loads)
            papers_df["authors"] = papers_df["authors"].apply(lambda x: x.split(", "))
            
            papers = papers_df.to_dict(orient="records")
            logger.info(f"Retrieved top {len(papers)} papers")
            return papers
            
        except Exception as e:
            logger.error(f"Error retrieving top papers: {str(e)}")
            return []
            
        finally:
            if 'conn' in locals():
                conn.close()
    
    def get_paper_by_id(self, paper_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific paper by ID.
        
        Args:
            paper_id: Paper ID
            
        Returns:
            Paper dictionary if found, None otherwise
        """
        try:
            conn = sqlite3.connect(self.db_path)
            
            query = "SELECT * FROM papers WHERE id = ?"
            papers_df = pd.read_sql_query(query, conn, params=(paper_id,))
            
            if len(papers_df) == 0:
                logger.warning(f"Paper not found: {paper_id}")
                return None
                
            # Process JSON fields
            papers_df["classification"] = papers_df["classification"].apply(json.loads)
            papers_df["novelty_assessment"] = papers_df["novelty_assessment"].apply(json.loads)
            papers_df["authors"] = papers_df["authors"].apply(lambda x: x.split(", "))
            
            paper = papers_df.iloc[0].to_dict()
            return paper
            
        except Exception as e:
            logger.error(f"Error retrieving paper: {str(e)}")
            return None
            
        finally:
            if 'conn' in locals():
                conn.close()
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get database statistics.
        
        Returns:
            Dictionary containing statistics
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get basic stats
            cursor.execute("SELECT COUNT(*), AVG(score), MAX(score) FROM papers")
            total_papers, avg_score, max_score = cursor.fetchone()
            
            # Get date range
            cursor.execute("""
            SELECT 
                MIN(DATE(processed_date)), 
                MAX(DATE(processed_date)),
                COUNT(DISTINCT DATE(processed_date))
            FROM papers
            """)
            first_date, last_date, total_days = cursor.fetchone()
            
            # Get category distribution
            cursor.execute("""
            SELECT classification, COUNT(*) as count 
            FROM papers 
            GROUP BY classification 
            ORDER BY count DESC
            LIMIT 5
            """)
            top_categories = [
                {"category": json.loads(row[0])["category"], "count": row[1]}
                for row in cursor.fetchall()
            ]
            
            return {
                "total_papers": total_papers or 0,
                "avg_score": round(avg_score, 2) if avg_score else 0,
                "max_score": round(max_score, 2) if max_score else 0,
                "first_date": first_date,
                "last_date": last_date,
                "total_days": total_days or 0,
                "top_categories": top_categories
            }
            
        except Exception as e:
            logger.error(f"Error getting statistics: {str(e)}")
            return {
                "total_papers": 0,
                "avg_score": 0,
                "max_score": 0,
                "first_date": None,
                "last_date": None,
                "total_days": 0,
                "top_categories": []
            }
            
        finally:
            if 'conn' in locals():
                conn.close()
