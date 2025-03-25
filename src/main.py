"""
LastWeek-MustRead-Arxiv main application.
"""

import os
import logging
import datetime
from typing import Dict, List, Optional
from dotenv import load_dotenv

from src.fetchers.arxiv_fetcher import ArxivFetcher
from src.agents.summarizer import SummarizerAgent
from src.agents.classifier import ClassifierAgent
from src.agents.novelty_assessor import NoveltyAssessorAgent
from src.agents.scorer import ScorerAgent
from src.database.db_manager import DatabaseManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class Config:
    """Application configuration."""
    
    def __init__(self, env_file: Optional[str] = None):
        """Initialize configuration."""
        if env_file:
            load_dotenv(env_file)
            
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.interested_fields = [
            "大型语言模型优化与效率",
            "多模态AI系统",
            "AI安全与对齐",
            "强化学习新方法",
            "生成式AI应用"
        ]
        self.arxiv_categories = ["cs.AI"]  # Can be expanded: ["cs.AI", "cs.LG", "cs.CL", "cs.CV", "cs.RO"]
        self.max_papers_per_run = 10
        self.database_path = "papers.db"
        self.llm_config = {
            "model": "gpt-4-turbo-preview",
            "temperature": 0.2,
        }

class PaperAnalyzer:
    """Main paper analysis orchestrator."""
    
    def __init__(self, config: Config):
        """Initialize the paper analyzer."""
        self.config = config
        self.fetcher = ArxivFetcher(max_results=config.max_papers_per_run)
        self.summarizer = SummarizerAgent(config.__dict__)
        self.classifier = ClassifierAgent(config.__dict__)
        self.novelty_assessor = NoveltyAssessorAgent(config.__dict__)
        self.scorer = ScorerAgent(config.__dict__)
        self.db_manager = DatabaseManager(config.database_path)
        
    def analyze_papers(self, days_range: int = 7) -> List[Dict]:
        """
        Analyze papers from the last N days.
        
        Args:
            days_range: Number of days to look back for papers
            
        Returns:
            List of analyzed paper results
        """
        logger.info(f"Starting paper analysis for the last {days_range} days")
        
        # Initialize database
        self.db_manager.initialize_database()
        
        # Fetch papers
        papers = self.fetcher.get_papers(
            categories=self.config.arxiv_categories,
            days_range=days_range,
            max_papers=self.config.max_papers_per_run
        )
        
        if not papers:
            logger.info("No papers found in the specified timeframe")
            return []
            
        logger.info(f"Found {len(papers)} papers to analyze")
        
        # Process each paper
        results = []
        for paper in papers:
            try:
                # Generate summary
                summary = self.summarizer.summarize_paper(paper)
                
                # Classify paper
                classification = self.classifier.classify_paper(paper, summary)
                
                # Assess novelty
                assessment = self.novelty_assessor.assess_novelty(paper, summary)
                
                # Score paper
                score, rationale = self.scorer.score_paper(
                    paper, summary, classification, assessment
                )
                
                # Create result
                result = {
                    "paper_id": paper["id"],
                    "title": paper["title"],
                    "authors": paper["authors"],
                    "published_date": paper["published"],
                    "processed_date": datetime.datetime.now().strftime("%Y-%m-%d"),
                    "summary": summary,
                    "classification": classification,
                    "novelty_assessment": assessment,
                    "score": score,
                    "scoring_rationale": rationale
                }
                
                # Save to database
                self.db_manager.save_paper_analysis(result)
                results.append(result)
                
                logger.info(f"Successfully analyzed paper: {paper['title']}")
                
            except Exception as e:
                logger.error(f"Error processing paper {paper['title']}: {str(e)}")
                continue
                
        return results

def main():
    """Main entry point."""
    config = Config()
    
    if not config.openai_api_key:
        logger.error("Please set the OPENAI_API_KEY environment variable")
        return
        
    analyzer = PaperAnalyzer(config)
    results = analyzer.analyze_papers(days_range=7)
    
    logger.info("\n=== Analysis Complete ===")
    logger.info(f"Processed {len(results)} papers")
    
    if results:
        # Print top 3 papers by score
        top_papers = sorted(results, key=lambda x: x["score"], reverse=True)[:3]
        logger.info("\nTop 3 Papers:")
        for i, paper in enumerate(top_papers, 1):
            logger.info(f"\n{i}. {paper['title']}")
            logger.info(f"Score: {paper['score']}/10")
            logger.info(f"Category: {paper['classification']['category']}")
            logger.info(f"Novelty: {paper['novelty_assessment']['score']}/10")

if __name__ == "__main__":
    main() 