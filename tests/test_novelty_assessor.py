"""Tests for the NoveltyAssessorAgent class."""

import pytest
from unittest.mock import Mock, patch
from src.agents.novelty_assessor import NoveltyAssessorAgent

@pytest.fixture
def sample_config():
    """Create a sample configuration for testing."""
    return {
        "openai_api_key": "test-key",
        "model": "gpt-4-turbo-preview",
        "temperature": 0.7
    }

@pytest.fixture
def sample_paper():
    """Create a sample paper for testing."""
    return {
        "title": "Test Paper Title",
        "authors": ["Author One", "Author Two"],
        "summary": "This is a test paper abstract about AI.",
        "text_content": """
        1. Introduction
        This is the introduction section of the paper.
        We propose a novel method for improving AI systems.
        
        2. Related Work
        Previous work has focused on traditional approaches.
        Our work builds upon these foundations.
        
        3. Methodology
        Our proposed method involves...
        """
    }

@pytest.fixture
def sample_summary():
    """Create a sample paper summary for testing."""
    return "The paper proposes a novel approach to improve AI systems."

@pytest.fixture
def mock_openai_response():
    """Create a mock OpenAI API response."""
    mock_response = Mock()
    mock_response.choices = [
        Mock(
            message=Mock(
                content="""
                ```json
                {
                    "score": 8.5,
                    "level": "Significant",
                    "description": "The paper demonstrates significant innovation...",
                    "strengths": ["Novel architecture", "Strong results"],
                    "limitations": ["High computational cost", "Limited testing"]
                }
                ```
                """
            )
        )
    ]
    return mock_response

def test_novelty_assessor_initialization(sample_config):
    """Test NoveltyAssessorAgent initialization."""
    agent = NoveltyAssessorAgent(sample_config)
    assert agent.model == "gpt-4-turbo-preview"
    assert agent.temperature == 0.7
    assert "expert AI research reviewer" in agent.system_message

def test_novelty_assessor_initialization_with_sample_data(sample_config):
    """Test NoveltyAssessorAgent initialization with sample data mode."""
    agent = NoveltyAssessorAgent(sample_config, use_sample_data=True)
    assert agent.use_sample_data is True
    assert not hasattr(agent, 'client')

def test_assess_novelty_with_sample_data(sample_config, sample_paper, sample_summary):
    """Test paper novelty assessment using sample data."""
    agent = NoveltyAssessorAgent(sample_config, use_sample_data=True)
    result = agent.assess_novelty(sample_paper, sample_summary)
    
    assert isinstance(result, dict)
    assert "score" in result
    assert "level" in result
    assert "description" in result
    assert "strengths" in result
    assert "limitations" in result
    assert 1 <= result["score"] <= 10
    assert result["level"] == "Significant"
    assert isinstance(result["strengths"], list)
    assert isinstance(result["limitations"], list)

@patch('openai.OpenAI')
def test_assess_novelty_with_api(mock_openai, sample_config, sample_paper, 
                                sample_summary, mock_openai_response):
    """Test paper novelty assessment using the OpenAI API."""
    # Setup mock
    mock_client = Mock()
    mock_client.chat.completions.create.return_value = mock_openai_response
    mock_openai.return_value = mock_client
    
    agent = NoveltyAssessorAgent(sample_config)
    result = agent.assess_novelty(sample_paper, sample_summary)
    
    assert result["score"] == 8.5
    assert result["level"] == "Significant"
    assert len(result["strengths"]) == 2
    assert len(result["limitations"]) == 2
    mock_client.chat.completions.create.assert_called_once()

@patch('openai.OpenAI')
def test_assess_novelty_api_error(mock_openai, sample_config, sample_paper, sample_summary):
    """Test error handling when API call fails."""
    # Setup mock to raise an exception
    mock_client = Mock()
    mock_client.chat.completions.create.side_effect = Exception("API Error")
    mock_openai.return_value = mock_client
    
    agent = NoveltyAssessorAgent(sample_config)
    
    with pytest.raises(RuntimeError) as exc_info:
        agent.assess_novelty(sample_paper, sample_summary)
    assert "Failed to assess paper novelty" in str(exc_info.value)

def test_build_novelty_prompt(sample_config, sample_paper, sample_summary):
    """Test prompt building functionality."""
    agent = NoveltyAssessorAgent(sample_config)
    prompt = agent._build_novelty_prompt(sample_paper, sample_summary)
    
    assert sample_paper["title"] in prompt
    assert sample_paper["summary"] in prompt
    assert sample_summary in prompt
    assert "Introduction Section" in prompt
    assert "Related Work Section" in prompt
    assert "novelty" in prompt.lower()
    assert "json" in prompt.lower()

def test_parse_novelty_result_valid_json(sample_config):
    """Test parsing of valid JSON novelty assessment result."""
    agent = NoveltyAssessorAgent(sample_config)
    test_result = """
    ```json
    {
        "score": 8.5,
        "level": "Significant",
        "description": "Test description",
        "strengths": ["Strength 1", "Strength 2"],
        "limitations": ["Limitation 1"]
    }
    ```
    """
    
    result = agent._parse_novelty_result(test_result)
    assert result["score"] == 8.5
    assert result["level"] == "Significant"
    assert len(result["strengths"]) == 2
    assert len(result["limitations"]) == 1

def test_parse_novelty_result_invalid_json(sample_config):
    """Test parsing of invalid JSON novelty assessment result."""
    agent = NoveltyAssessorAgent(sample_config)
    test_result = "Invalid JSON content"
    
    with pytest.raises(ValueError) as exc_info:
        agent._parse_novelty_result(test_result)
    assert "Invalid JSON format" in str(exc_info.value)

def test_parse_novelty_result_missing_fields(sample_config):
    """Test parsing of JSON result with missing required fields."""
    agent = NoveltyAssessorAgent(sample_config)
    test_result = """
    ```json
    {
        "score": 8.5,
        "level": "Significant"
    }
    ```
    """
    
    with pytest.raises(ValueError) as exc_info:
        agent._parse_novelty_result(test_result)
    assert "Missing required fields" in str(exc_info.value)

def test_parse_novelty_result_invalid_score(sample_config):
    """Test parsing of JSON result with invalid score value."""
    agent = NoveltyAssessorAgent(sample_config)
    test_result = """
    ```json
    {
        "score": 11.0,
        "level": "Significant",
        "description": "Test description",
        "strengths": ["Strength 1"],
        "limitations": ["Limitation 1"]
    }
    ```
    """
    
    with pytest.raises(ValueError) as exc_info:
        agent._parse_novelty_result(test_result)
    assert "Invalid score value" in str(exc_info.value)

def test_parse_novelty_result_invalid_lists(sample_config):
    """Test parsing of JSON result with invalid list fields."""
    agent = NoveltyAssessorAgent(sample_config)
    test_result = """
    ```json
    {
        "score": 8.5,
        "level": "Significant",
        "description": "Test description",
        "strengths": "Not a list",
        "limitations": ["Limitation 1"]
    }
    ```
    """
    
    with pytest.raises(ValueError) as exc_info:
        agent._parse_novelty_result(test_result)
    assert "Strengths must be a list" in str(exc_info.value)

def test_extract_introduction_section(sample_config):
    """Test extraction of introduction section from paper text."""
    agent = NoveltyAssessorAgent(sample_config)
    test_text = """
    1. Introduction
    This is the introduction.
    It contains important information.
    
    2. Related Work
    This is the related work section.
    """
    
    intro = agent._extract_introduction_section(test_text)
    assert "This is the introduction" in intro
    assert "Related Work" not in intro

def test_extract_related_work_section(sample_config):
    """Test extraction of related work section from paper text."""
    agent = NoveltyAssessorAgent(sample_config)
    test_text = """
    1. Introduction
    This is the introduction.
    
    2. Related Work
    This is the related work section.
    It discusses previous research.
    
    3. Methodology
    This is the methodology section.
    """
    
    related_work = agent._extract_related_work_section(test_text)
    assert "This is the related work section" in related_work
    assert "Introduction" not in related_work
    assert "Methodology" not in related_work

def test_extract_sections_empty_text(sample_config):
    """Test section extraction with empty text."""
    agent = NoveltyAssessorAgent(sample_config)
    assert agent._extract_introduction_section("") == ""
    assert agent._extract_related_work_section("") == ""

def test_extract_sections_no_clear_sections(sample_config):
    """Test section extraction with text that has no clear section markers."""
    agent = NoveltyAssessorAgent(sample_config)
    test_text = "This is just a continuous block of text without clear sections."
    
    intro = agent._extract_introduction_section(test_text)
    related_work = agent._extract_related_work_section(test_text)
    
    assert intro == test_text[:1000].strip()
    assert related_work == "" 