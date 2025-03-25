"""Tests for the ScorerAgent class."""

import pytest
from unittest.mock import Mock, patch
from src.agents.scorer import ScorerAgent

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
        "text_content": "This is the main content of the paper. We propose a new method..."
    }

@pytest.fixture
def sample_summary():
    """Create a sample paper summary for testing."""
    return "The paper proposes a novel approach to improve language model performance."

@pytest.fixture
def sample_classification():
    """Create a sample classification result for testing."""
    return {
        "category": "Language Models",
        "rationale": "The paper focuses on improving language model capabilities.",
        "confidence": 0.9
    }

@pytest.fixture
def sample_novelty():
    """Create a sample novelty assessment for testing."""
    return {
        "score": 8.0,
        "level": "Significant",
        "description": "The proposed method shows significant improvements.",
        "strengths": ["Novel architecture", "Strong results"],
        "limitations": ["High computational cost", "Limited testing"]
    }

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
                    "rationale": "The paper demonstrates strong innovation...",
                    "breakdown": {
                        "innovation": 8.0,
                        "technical_depth": 8.5,
                        "experimental_quality": 8.5,
                        "potential_impact": 9.0,
                        "practical_value": 8.5
                    }
                }
                ```
                """
            )
        )
    ]
    return mock_response

def test_scorer_initialization(sample_config):
    """Test ScorerAgent initialization."""
    agent = ScorerAgent(sample_config)
    assert agent.model == "gpt-4-turbo-preview"
    assert agent.temperature == 0.7
    assert "expert AI research evaluation specialist" in agent.system_message

def test_scorer_initialization_with_sample_data(sample_config):
    """Test ScorerAgent initialization with sample data mode."""
    agent = ScorerAgent(sample_config, use_sample_data=True)
    assert agent.use_sample_data is True
    assert not hasattr(agent, 'client')

def test_score_paper_with_sample_data(sample_config, sample_paper, sample_summary, 
                                    sample_classification, sample_novelty):
    """Test paper scoring using sample data."""
    agent = ScorerAgent(sample_config, use_sample_data=True)
    score, results = agent.score_paper(sample_paper, sample_summary, 
                                     sample_classification, sample_novelty)
    
    assert isinstance(score, float)
    assert 0 <= score <= 10
    assert isinstance(results, dict)
    assert "score" in results
    assert "rationale" in results
    assert "breakdown" in results
    assert all(key in results["breakdown"] for key in [
        "innovation", "technical_depth", "experimental_quality", 
        "potential_impact", "practical_value"
    ])

@patch('openai.OpenAI')
def test_score_paper_with_api(mock_openai, sample_config, sample_paper, sample_summary,
                             sample_classification, sample_novelty, mock_openai_response):
    """Test paper scoring using the OpenAI API."""
    # Setup mock
    mock_client = Mock()
    mock_client.chat.completions.create.return_value = mock_openai_response
    mock_openai.return_value = mock_client
    
    agent = ScorerAgent(sample_config)
    score, results = agent.score_paper(sample_paper, sample_summary, 
                                     sample_classification, sample_novelty)
    
    assert score == 8.5
    assert isinstance(results, dict)
    assert results["score"] == 8.5
    assert "rationale" in results
    assert "breakdown" in results
    assert results["breakdown"]["innovation"] == 8.0
    mock_client.chat.completions.create.assert_called_once()

@patch('openai.OpenAI')
def test_score_paper_api_error(mock_openai, sample_config, sample_paper, sample_summary,
                              sample_classification, sample_novelty):
    """Test error handling when API call fails."""
    # Setup mock to raise an exception
    mock_client = Mock()
    mock_client.chat.completions.create.side_effect = Exception("API Error")
    mock_openai.return_value = mock_client
    
    agent = ScorerAgent(sample_config)
    
    with pytest.raises(RuntimeError) as exc_info:
        agent.score_paper(sample_paper, sample_summary, sample_classification, sample_novelty)
    assert "Failed to score paper" in str(exc_info.value)

def test_build_scoring_prompt(sample_config, sample_paper, sample_summary,
                            sample_classification, sample_novelty):
    """Test prompt building functionality."""
    agent = ScorerAgent(sample_config)
    prompt = agent._build_scoring_prompt(sample_paper, sample_summary,
                                       sample_classification, sample_novelty)
    
    assert sample_paper["title"] in prompt
    assert sample_paper["summary"] in prompt
    assert sample_summary in prompt
    assert sample_classification["category"] in prompt
    assert sample_novelty["description"] in prompt
    assert "innovation" in prompt.lower()
    assert "technical depth" in prompt.lower()
    assert "experimental quality" in prompt.lower()

def test_parse_scoring_result_valid_json(sample_config):
    """Test parsing of valid JSON scoring result."""
    agent = ScorerAgent(sample_config)
    test_result = """
    ```json
    {
        "score": 8.5,
        "rationale": "Test rationale",
        "breakdown": {
            "innovation": 8.0,
            "technical_depth": 8.5,
            "experimental_quality": 8.5,
            "potential_impact": 9.0,
            "practical_value": 8.5
        }
    }
    ```
    """
    
    score, results = agent._parse_scoring_result(test_result)
    assert score == 8.5
    assert results["rationale"] == "Test rationale"
    assert results["breakdown"]["innovation"] == 8.0

def test_parse_scoring_result_invalid_json(sample_config):
    """Test parsing of invalid JSON scoring result."""
    agent = ScorerAgent(sample_config)
    test_result = "Invalid JSON content"
    
    with pytest.raises(ValueError) as exc_info:
        agent._parse_scoring_result(test_result)
    assert "Invalid JSON format" in str(exc_info.value)

def test_parse_scoring_result_missing_fields(sample_config):
    """Test parsing of JSON result with missing required fields."""
    agent = ScorerAgent(sample_config)
    test_result = """
    ```json
    {
        "score": 8.5
    }
    ```
    """
    
    with pytest.raises(ValueError) as exc_info:
        agent._parse_scoring_result(test_result)
    assert "Missing required fields" in str(exc_info.value)

def test_parse_scoring_result_invalid_score(sample_config):
    """Test parsing of JSON result with invalid score value."""
    agent = ScorerAgent(sample_config)
    test_result = """
    ```json
    {
        "score": 11.0,
        "rationale": "Test rationale"
    }
    ```
    """
    
    with pytest.raises(ValueError) as exc_info:
        agent._parse_scoring_result(test_result)
    assert "Invalid score value" in str(exc_info.value) 