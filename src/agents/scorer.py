# agents/scorer.py
"""Agent for scoring research papers based on their academic value and potential impact."""

import logging
import json
import re
from typing import Dict, Any, Tuple, Optional
from openai import OpenAI
from src.utils.sample_data import get_sample_papers

logger = logging.getLogger(__name__)

class ScorerAgent:
    """Agent responsible for evaluating and scoring research papers.
    
    This agent uses OpenAI's API to analyze papers and provide a comprehensive score
    based on multiple factors including innovation, technical depth, and potential impact.
    """
    
    def __init__(self, config: Dict[str, Any], use_sample_data: bool = False):
        """Initialize the ScorerAgent.
        
        Args:
            config: Configuration dictionary containing OpenAI API settings
            use_sample_data: If True, use sample data instead of making API calls
        """
        self.config = config
        self.use_sample_data = use_sample_data
        
        # Initialize OpenAI client
        if not use_sample_data:
            self.client = OpenAI(api_key=config.get("openai_api_key"))
            self.model = config.get("model", "gpt-4-turbo-preview")
            self.temperature = config.get("temperature", 0.7)
        
        self.system_message = """
        You are an expert AI research evaluation specialist, skilled at assessing papers' 
        academic value and potential impact.
        
        Your task is to score papers (0-10) based on their innovation, technical depth, 
        practical value, and research significance.
        
        Consider the following factors in your evaluation:
        - Innovation: Novelty and uniqueness of the method
        - Technical Depth: Technical complexity and theoretical foundation
        - Experimental Quality: Rigor of experiments and convincing results
        - Potential Impact: Potential contribution to field development
        - Practical Value: Potential for real-world applications
        
        Provide detailed rationale for your scores, explaining strengths, weaknesses, 
        and how you weighted different factors.
        """
    
    def score_paper(self, paper: Dict[str, Any], summary: str, 
                   classification: Dict[str, Any], novelty: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """Score a paper based on its academic value.
        
        Args:
            paper: Dictionary containing paper information
            summary: Paper's main contributions summary
            classification: Paper's classification results
            novelty: Paper's novelty assessment results
            
        Returns:
            Tuple of (score, detailed_results)
        """
        if self.use_sample_data:
            logger.info(f"Using sample data for paper: {paper.get('title', 'Unknown')}")
            return 8.5, {
                "score": 8.5,
                "rationale": "Sample scoring rationale",
                "breakdown": {
                    "innovation": 8.0,
                    "technical_depth": 8.5,
                    "experimental_quality": 8.5,
                    "potential_impact": 9.0,
                    "practical_value": 8.5
                }
            }
            
        logger.info(f"Scoring paper: {paper.get('title', 'Unknown')}")
        
        try:
            prompt = self._build_scoring_prompt(paper, summary, classification, novelty)
            
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                messages=[
                    {"role": "system", "content": self.system_message},
                    {"role": "user", "content": prompt}
                ]
            )
            
            result = response.choices[0].message.content
            score, detailed_results = self._parse_scoring_result(result)
            
            logger.info(f"Successfully scored paper: {score}/10")
            return score, detailed_results
            
        except Exception as e:
            logger.error(f"Error scoring paper: {str(e)}")
            raise RuntimeError(f"Failed to score paper: {str(e)}")
    
    def _build_scoring_prompt(self, paper: Dict[str, Any], summary: str, 
                            classification: Dict[str, Any], novelty: Dict[str, Any]) -> str:
        """Build the scoring prompt for the paper.
        
        Args:
            paper: Dictionary containing paper information
            summary: Paper's main contributions summary
            classification: Paper's classification results
            novelty: Paper's novelty assessment results
            
        Returns:
            Formatted prompt string
        """
        title = paper.get("title", "")
        abstract = paper.get("summary", "")
        
        prompt = f"""
        Please evaluate and score the following AI research paper (0-10).
        
        Paper Information:
        Title: {title}
        Abstract: {abstract}
        
        Analysis:
        1. Main Contributions: {summary}
        
        2. Research Area: {classification.get('category', 'Unknown')}
           Classification Rationale: {classification.get('rationale', 'None')}
        
        3. Novelty Assessment:
           Score: {novelty.get('score', 'N/A')}/10
           Level: {novelty.get('level', 'N/A')}
           Description: {novelty.get('description', 'N/A')}
           Strengths: {', '.join(novelty.get('strengths', []))}
           Limitations: {', '.join(novelty.get('limitations', []))}
        
        Please consider the above information and your expert judgment to score the paper (0-10),
        providing detailed rationale.
        
        Consider these factors in your evaluation:
        - Innovation: Novelty and uniqueness of the method
        - Technical Depth: Technical complexity and theoretical foundation
        - Experimental Quality: Rigor of experiments and convincing results
        - Potential Impact: Potential contribution to field development
        - Practical Value: Potential for real-world applications
        
        Provide your evaluation in the following JSON format:
        ```json
        {{
            "score": 7.5,
            "rationale": "Detailed scoring rationale...",
            "breakdown": {{
                "innovation": 8.0,
                "technical_depth": 7.0,
                "experimental_quality": 7.5,
                "potential_impact": 8.0,
                "practical_value": 7.0
            }}
        }}
        ```
        
        Return only the JSON result without additional explanation.
        """
        return prompt
    
    def _parse_scoring_result(self, result: str) -> Tuple[float, Dict[str, Any]]:
        """Parse the scoring result from the API response.
        
        Args:
            result: LLM response text
            
        Returns:
            Tuple of (score, detailed_results)
            
        Raises:
            ValueError: If unable to parse the scoring result
        """
        try:
            # Try to extract JSON content from code block
            json_match = re.search(r'```(?:json)?(.*?)```', result, re.DOTALL)
            if json_match:
                json_str = json_match.group(1).strip()
            else:
                # If no code block, try to parse the entire string as JSON
                json_str = result.strip()
            
            scoring = json.loads(json_str)
            
            # Validate required fields
            if "score" not in scoring or "rationale" not in scoring:
                raise ValueError("Missing required fields in scoring result")
            
            score = float(scoring["score"])
            if not 0 <= score <= 10:
                raise ValueError(f"Invalid score value: {score}")
            
            # Clean up and validate breakdown scores if present
            if "breakdown" in scoring:
                breakdown = scoring["breakdown"]
                for category, sub_score in breakdown.items():
                    if not isinstance(sub_score, (int, float)) or not 0 <= sub_score <= 10:
                        raise ValueError(f"Invalid breakdown score for {category}: {sub_score}")
            
            return score, scoring
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON result: {str(e)}")
            raise ValueError(f"Invalid JSON format in scoring result: {str(e)}")
        except Exception as e:
            logger.error(f"Error parsing scoring result: {str(e)}")
            raise ValueError(f"Failed to parse scoring result: {str(e)}")
