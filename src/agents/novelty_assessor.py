# agents/novelty_assessor.py
import logging
import json
import re
from typing import Dict, Any, List, Optional, Tuple
from openai import OpenAI
from src.utils.sample_data import get_sample_papers

logger = logging.getLogger(__name__)

class NoveltyAssessorAgent:
    """Agent responsible for evaluating paper novelty and incremental contributions.
    
    This agent uses OpenAI's API to analyze papers and assess their novelty level,
    focusing on innovation, technical contributions, and improvements over existing work.
    """
    
    def __init__(self, config: Dict[str, Any], use_sample_data: bool = False):
        """Initialize the NoveltyAssessorAgent.
        
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
        You are an expert AI research reviewer, skilled at assessing paper novelty 
        and incremental contributions.
        
        Analyze paper content, especially introduction, related work, and methodology 
        sections, to determine the level of innovation.
        
        Consider these aspects:
        1. Does the paper propose a new method or improve existing ones?
        2. How significant are the improvements over existing work?
        3. Is the technical approach unique or innovative?
        4. How challenging and important is the problem being solved?
        
        Look for phrases indicating novelty like "first to propose", "breakthrough work",
        "significantly outperforms", while also noting any limitations acknowledged by
        the authors.
        
        Provide a novelty score (1-10) and detailed assessment rationale.
        """
    
    def assess_novelty(self, paper: Dict[str, Any], summary: str) -> Dict[str, Any]:
        """Assess the novelty and incremental contributions of a paper.
        
        Args:
            paper: Dictionary containing paper information
            summary: Paper's main contributions summary
            
        Returns:
            Dictionary containing novelty assessment results
            
        Raises:
            RuntimeError: If assessment fails
        """
        if self.use_sample_data:
            logger.info(f"Using sample data for paper: {paper.get('title', 'Unknown')}")
            return {
                "score": 8.5,
                "level": "Significant",
                "description": "Sample novelty assessment",
                "strengths": ["Sample strength 1", "Sample strength 2"],
                "limitations": ["Sample limitation 1"]
            }
            
        logger.info(f"Assessing novelty for paper: {paper.get('title', 'Unknown')}")
        
        try:
            prompt = self._build_novelty_prompt(paper, summary)
            
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                messages=[
                    {"role": "system", "content": self.system_message},
                    {"role": "user", "content": prompt}
                ]
            )
            
            result = response.choices[0].message.content
            assessment = self._parse_novelty_result(result)
            
            logger.info(f"Successfully assessed paper novelty: {assessment['score']}/10")
            return assessment
            
        except Exception as e:
            logger.error(f"Error assessing paper novelty: {str(e)}")
            raise RuntimeError(f"Failed to assess paper novelty: {str(e)}")
    
    def _build_novelty_prompt(self, paper: Dict[str, Any], summary: str) -> str:
        """Build the novelty assessment prompt for the paper.
        
        Args:
            paper: Dictionary containing paper information
            summary: Paper's main contributions summary
            
        Returns:
            Formatted prompt string
        """
        title = paper.get("title", "")
        abstract = paper.get("summary", "")
        
        # Extract key sections from full text if available
        text_content = paper.get("text_content", "")
        intro_section = self._extract_introduction_section(text_content)
        related_work_section = self._extract_related_work_section(text_content)
        
        prompt = f"""
        Please assess the novelty and incremental contributions of the following AI paper.
        
        Paper Information:
        Title: {title}
        Abstract: {abstract}
        
        Main Contributions:
        {summary}
        
        Introduction Section:
        {intro_section}
        
        Related Work Section:
        {related_work_section}
        
        Evaluate the paper's novelty relative to existing work. Focus on:
        1. Does it propose a new method or improve existing ones?
        2. How significant are the improvements? Revolutionary or incremental?
        3. Where does the innovation lie - algorithms, models, applications, or theory?
        4. Does it solve important challenges or open new research directions?
        
        Provide your assessment in the following JSON format:
        ```json
        {{
            "score": 7.5,  # Novelty score, 1-10
            "level": "Significant",  # Novelty level: Low, Moderate, Significant, Breakthrough
            "description": "Detailed assessment...",
            "strengths": ["Innovation 1", "Innovation 2"...],
            "limitations": ["Limitation 1", "Limitation 2"...]
        }}
        ```
        
        Return only the JSON result without additional explanation.
        """
        return prompt
    
    def _parse_novelty_result(self, result: str) -> Dict[str, Any]:
        """Parse the novelty assessment result from the API response.
        
        Args:
            result: LLM response text
            
        Returns:
            Parsed assessment result dictionary
            
        Raises:
            ValueError: If unable to parse the assessment result
        """
        try:
            # Try to extract JSON content from code block
            json_match = re.search(r'```(?:json)?(.*?)```', result, re.DOTALL)
            if json_match:
                json_str = json_match.group(1).strip()
            else:
                # If no code block, try to parse the entire string as JSON
                json_str = result.strip()
            
            assessment = json.loads(json_str)
            
            # Validate required fields
            required_fields = ["score", "level", "description", "strengths", "limitations"]
            if not all(key in assessment for key in required_fields):
                raise ValueError("Missing required fields in assessment result")
            
            # Validate score
            score = float(assessment["score"])
            if not 1 <= score <= 10:
                raise ValueError(f"Invalid score value: {score}")
            
            # Validate level
            valid_levels = ["Low", "Moderate", "Significant", "Breakthrough"]
            if assessment["level"] not in valid_levels:
                logger.warning(f"Unexpected novelty level: {assessment['level']}")
            
            # Validate lists
            if not isinstance(assessment["strengths"], list):
                raise ValueError("Strengths must be a list")
            if not isinstance(assessment["limitations"], list):
                raise ValueError("Limitations must be a list")
            
            return assessment
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON result: {str(e)}")
            raise ValueError(f"Invalid JSON format in assessment result: {str(e)}")
        except Exception as e:
            logger.error(f"Error parsing assessment result: {str(e)}")
            raise ValueError(f"Failed to parse assessment result: {str(e)}")
    
    def _extract_introduction_section(self, text_content: str) -> str:
        """Extract the introduction section from the paper's full text.
        
        Args:
            text_content: Full text content of the paper
            
        Returns:
            Extracted introduction section or empty string if not found
        """
        if not text_content:
            return ""
            
        # Common section headers
        intro_patterns = [
            r"(?i)1\.?\s*Introduction\s*\n",
            r"(?i)I\.?\s*Introduction\s*\n",
            r"(?i)\n[^\n]*Introduction[^\n]*\n"
        ]
        
        next_section_patterns = [
            r"(?i)2\.?\s*[A-Z]",
            r"(?i)II\.?\s*[A-Z]",
            r"(?i)\n[^\n]*Related Work[^\n]*\n",
            r"(?i)\n[^\n]*Background[^\n]*\n",
            r"(?i)\n[^\n]*Methodology[^\n]*\n"
        ]
        
        # Try to find the introduction section
        for pattern in intro_patterns:
            match = re.search(pattern, text_content)
            if match:
                start = match.end()
                # Find the start of the next section
                end = len(text_content)
                for next_pattern in next_section_patterns:
                    next_match = re.search(next_pattern, text_content[start:])
                    if next_match:
                        end = start + next_match.start()
                        break
                return text_content[start:end].strip()
        
        # If no clear introduction section found, return first 1000 characters
        return text_content[:1000].strip()
    
    def _extract_related_work_section(self, text_content: str) -> str:
        """Extract the related work section from the paper's full text.
        
        Args:
            text_content: Full text content of the paper
            
        Returns:
            Extracted related work section or empty string if not found
        """
        if not text_content:
            return ""
            
        # Common section headers
        related_work_patterns = [
            r"(?i)2\.?\s*Related Work\s*\n",
            r"(?i)II\.?\s*Related Work\s*\n",
            r"(?i)\n[^\n]*Related Work[^\n]*\n",
            r"(?i)\n[^\n]*Previous Work[^\n]*\n",
            r"(?i)\n[^\n]*Background[^\n]*\n"
        ]
        
        next_section_patterns = [
            r"(?i)3\.?\s*[A-Z]",
            r"(?i)III\.?\s*[A-Z]",
            r"(?i)\n[^\n]*Methodology[^\n]*\n",
            r"(?i)\n[^\n]*Proposed Method[^\n]*\n",
            r"(?i)\n[^\n]*Approach[^\n]*\n"
        ]
        
        # Try to find the related work section
        for pattern in related_work_patterns:
            match = re.search(pattern, text_content)
            if match:
                start = match.end()
                # Find the start of the next section
                end = len(text_content)
                for next_pattern in next_section_patterns:
                    next_match = re.search(next_pattern, text_content[start:])
                    if next_match:
                        end = start + next_match.start()
                        break
                return text_content[start:end].strip()
        
        return ""
