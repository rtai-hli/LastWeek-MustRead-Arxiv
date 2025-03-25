# agents/summarizer.py
"""Agent for extracting key contributions and innovations from research papers."""

import logging
from typing import Dict, Any, Optional
import openai
from openai import OpenAI
from src.utils.sample_data import get_sample_papers

logger = logging.getLogger(__name__)

class SummarizerAgent:
    """Agent responsible for extracting main contributions and innovations from research papers.
    
    This agent uses OpenAI's API to generate comprehensive summaries of research papers,
    focusing on their key contributions, methodologies, and potential impact.
    """
    
    def __init__(self, config: Dict[str, Any], use_sample_data: bool = False):
        """Initialize the SummarizerAgent.
        
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
        You are an expert AI paper summarization specialist, skilled at extracting key research 
        contributions and innovations from papers. Your task is to read the paper content and 
        extract the main contributions, innovative methods, and key findings.
        
        Focus on phrases like "we propose...", "contributions include...", "our innovations are..."
        Keep the summary concise, accurate, and comprehensive, highlighting the core innovations 
        and academic value.
        """
    
    def summarize_paper(self, paper: Dict[str, Any]) -> Dict[str, str]:
        """Generate a summary of the paper's main contributions.
        
        Args:
            paper: Dictionary containing paper information including title, authors, 
                  abstract, and full text
            
        Returns:
            Dictionary containing structured summary sections
        """
        if self.use_sample_data:
            logger.info(f"Using sample data for paper: {paper.get('title', 'Unknown')}")
            sample_papers = get_sample_papers()
            # Return a pre-written summary for sample data
            return {
                "research_problem": "Sample research problem",
                "methodology": "Sample methodology",
                "innovations": "Sample innovations",
                "findings": "Sample findings",
                "impact": "Sample impact"
            }
            
        logger.info(f"Generating summary for paper: {paper.get('title', 'Unknown')}")
        
        try:
            prompt = self._build_summarization_prompt(paper)
            
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                messages=[
                    {"role": "system", "content": self.system_message},
                    {"role": "user", "content": prompt}
                ]
            )
            
            summary = response.choices[0].message.content
            logger.info(f"Successfully generated summary for: {paper.get('title', 'Unknown')}")
            
            # Parse the structured summary into sections
            sections = self._parse_summary_sections(summary)
            return sections
            
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            raise RuntimeError(f"Failed to generate summary: {str(e)}")
    
    def _build_summarization_prompt(self, paper: Dict[str, Any]) -> str:
        """Build the summarization prompt for the paper.
        
        Args:
            paper: Dictionary containing paper information
            
        Returns:
            Formatted prompt string
        """
        title = paper.get("title", "")
        authors = paper.get("authors", [])
        abstract = paper.get("summary", "")
        
        # If full text exists, take first 5000 and last 2000 chars (intro and conclusion)
        text_content = paper.get("text_content", "")
        if len(text_content) > 7000:
            text_sample = text_content[:5000] + "\n...\n" + text_content[-2000:]
        else:
            text_sample = text_content
        
        prompt = f"""
        Please analyze the following AI research paper and extract its main contributions, 
        innovative methods, and key findings.
        
        Title: {title}
        Authors: {', '.join(authors) if isinstance(authors, list) else authors}
        Abstract: {abstract}
        
        Paper Content:
        {text_sample}
        
        Please provide a concise but comprehensive summary with the following structure:
        
        1. Research Problem: [Describe the main problem addressed]
        2. Methodology: [Outline the proposed methods or techniques]
        3. Key Innovations: [List the main innovations and contributions]
        4. Findings/Results: [Summarize key findings and experimental results]
        5. Potential Impact: [Analyze potential impact on the AI field]
        
        Focus on unique contributions rather than general descriptions. Pay special attention 
        to key phrases indicating innovation like "our main contributions...", "we propose...", 
        "compared to existing methods..."
        """
        return prompt
    
    def _parse_summary_sections(self, summary: str) -> Dict[str, str]:
        """Parse the generated summary into structured sections.
        
        Args:
            summary: Raw summary text from the API
            
        Returns:
            Dictionary containing structured summary sections
        """
        sections = {
            "research_problem": "",
            "methodology": "",
            "innovations": "",
            "findings": "",
            "impact": ""
        }
        
        current_section = None
        lines = summary.split("\n")
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if "Research Problem:" in line:
                current_section = "research_problem"
            elif "Methodology:" in line or "Main Methods:" in line:
                current_section = "methodology"
            elif "Key Innovations:" in line or "Core Innovations:" in line:
                current_section = "innovations"
            elif "Findings" in line or "Results:" in line:
                current_section = "findings"
            elif "Impact:" in line or "Potential Impact:" in line:
                current_section = "impact"
            elif current_section:
                sections[current_section] += line + " "
        
        # Clean up the sections
        for key in sections:
            sections[key] = sections[key].strip()
        
        return sections
