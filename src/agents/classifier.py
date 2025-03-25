# agents/classifier.py
"""Agent for classifying research papers into predefined research areas."""

import logging
import json
import re
from typing import Dict, Any, List, Optional
from openai import OpenAI
from src.utils.sample_data import get_sample_papers

logger = logging.getLogger(__name__)

class ClassifierAgent:
    """Agent responsible for classifying papers into predefined research areas.
    
    This agent uses OpenAI's API to analyze papers and classify them into one of several
    predefined research areas, providing confidence scores and rationale for each classification.
    """
    
    def __init__(self, config: Dict[str, Any], use_sample_data: bool = False):
        """Initialize the ClassifierAgent.
        
        Args:
            config: Configuration dictionary containing OpenAI API settings and interested fields
            use_sample_data: If True, use sample data instead of making API calls
        """
        self.config = config
        self.use_sample_data = use_sample_data
        self.interested_fields = config.get("interested_fields", [
            "Large Language Models",
            "Computer Vision",
            "Reinforcement Learning",
            "Neural Architecture",
            "AI Safety"
        ])
        
        # Initialize OpenAI client
        if not use_sample_data:
            self.client = OpenAI(api_key=config.get("openai_api_key"))
            self.model = config.get("model", "gpt-4-turbo-preview")
            self.temperature = config.get("temperature", 0.7)
        
        self.system_message = f"""
        You are an expert AI paper classification specialist, skilled at categorizing papers 
        into specific research areas.
        
        You need to classify papers into one of the following areas:
        {', '.join(self.interested_fields)}
        
        If a paper spans multiple areas, choose the most prominent one.
        If a paper doesn't fit any of these areas, classify it as "Other".
        
        Provide a detailed rationale for each classification decision.
        """
    
    def classify_paper(self, paper: Dict[str, Any], summary: str) -> Dict[str, Any]:
        """Classify a paper into one of the interested research areas.
        
        Args:
            paper: Dictionary containing paper information
            summary: Paper's main contributions summary
            
        Returns:
            Dictionary containing classification results and rationale
            
        Raises:
            RuntimeError: If classification fails
        """
        if self.use_sample_data:
            logger.info(f"Using sample data for paper: {paper.get('title', 'Unknown')}")
            return {
                "category": "Large Language Models",
                "confidence": 0.9,
                "rationale": "Sample classification rationale"
            }
            
        logger.info(f"Classifying paper: {paper.get('title', 'Unknown')}")
        
        try:
            prompt = self._build_classification_prompt(paper, summary)
            
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                messages=[
                    {"role": "system", "content": self.system_message},
                    {"role": "user", "content": prompt}
                ]
            )
            
            result = response.choices[0].message.content
            classification = self._parse_classification_result(result)
            
            logger.info(f"Successfully classified paper as: {classification['category']}")
            return classification
            
        except Exception as e:
            logger.error(f"Error classifying paper: {str(e)}")
            raise RuntimeError(f"Failed to classify paper: {str(e)}")
    
    def _build_classification_prompt(self, paper: Dict[str, Any], summary: str) -> str:
        """Build the classification prompt for the paper.
        
        Args:
            paper: Dictionary containing paper information
            summary: Paper's main contributions summary
            
        Returns:
            Formatted prompt string
        """
        title = paper.get("title", "")
        abstract = paper.get("summary", "")
        
        prompt = f"""
        Please classify the following AI research paper into one of our areas of interest.
        
        Available Research Areas:
        {', '.join([f"{i+1}. {field}" for i, field in enumerate(self.interested_fields)])}
        
        If the paper doesn't fit any of these areas, classify it as "Other".
        
        Paper Information:
        Title: {title}
        Abstract: {abstract}
        
        Main Contributions:
        {summary}
        
        Provide your classification in the following JSON format:
        ```json
        {{
            "category": "chosen_area_name",
            "confidence": 0.85,  # Classification confidence, float between 0-1
            "rationale": "Detailed explanation of classification reasoning..."
        }}
        ```
        
        Return only the JSON result without additional explanation.
        """
        return prompt
    
    def _parse_classification_result(self, result: str) -> Dict[str, Any]:
        """Parse the classification result from the API response.
        
        Args:
            result: LLM response text
            
        Returns:
            Parsed classification result dictionary
            
        Raises:
            ValueError: If unable to parse the classification result
        """
        try:
            # Try to extract JSON content from code block
            json_match = re.search(r'```(?:json)?(.*?)```', result, re.DOTALL)
            if json_match:
                json_str = json_match.group(1).strip()
            else:
                # If no code block, try to parse the entire string as JSON
                json_str = result.strip()
            
            classification = json.loads(json_str)
            
            # Validate required fields
            if not all(key in classification for key in ["category", "confidence", "rationale"]):
                raise ValueError("Missing required fields in classification result")
            
            # Validate confidence score
            confidence = float(classification["confidence"])
            if not 0 <= confidence <= 1:
                raise ValueError(f"Invalid confidence value: {confidence}")
            
            # Validate category
            category = classification["category"]
            if category not in self.interested_fields and category != "Other":
                logger.warning(f"Unexpected category: {category}")
            
            return classification
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON result: {str(e)}")
            raise ValueError(f"Invalid JSON format in classification result: {str(e)}")
        except Exception as e:
            logger.error(f"Error parsing classification result: {str(e)}")
            raise ValueError(f"Failed to parse classification result: {str(e)}")
