"""Tests for the ClassifierAgent class."""

import pytest
from unittest.mock import Mock, patch
from src.agents.classifier import ClassifierAgent

@pytest.fixture
def sample_config():
    """Create a sample configuration for testing."""
    return {
        "openai_api_key": "test-key",
        "model": "gpt-4-turbo-preview",
        "temperature": 0.7,
        "interested_fields": [
            "Large Language Models",
            "Computer Vision",
            "Reinforcement Learning",
            "Neural Architecture",
            "AI Safety"
        ]
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
def mock_openai_response():
    """Create a mock OpenAI API response."""
    mock_response = Mock()
    mock_response.choices = [
        Mock(
            message=Mock(
                content="""
                ```json
                {
                    "category": "Large Language Models",
                    "confidence": 0.9,
                    "rationale": "The paper focuses on language model improvements..."
                }
                ```
                """
            )
        )
    ]
    return mock_response

def test_classifier_initialization(sample_config):
    """Test ClassifierAgent initialization."""
    agent = ClassifierAgent(sample_config)
    assert agent.model == "gpt-4-turbo-preview"
    assert agent.temperature == 0.7
    assert len(agent.interested_fields) == 5
    assert "expert AI paper classification specialist" in agent.system_message

def test_classifier_initialization_with_sample_data(sample_config):
    """Test ClassifierAgent initialization with sample data mode."""
    agent = ClassifierAgent(sample_config, use_sample_data=True)
    assert agent.use_sample_data is True
    assert not hasattr(agent, 'client')
    assert len(agent.interested_fields) == 5

def test_classifier_initialization_default_fields():
    """Test ClassifierAgent initialization with default fields."""
    agent = ClassifierAgent({})
    assert len(agent.interested_fields) == 5
    assert "Large Language Models" in agent.interested_fields
    assert "Computer Vision" in agent.interested_fields

def test_classify_paper_with_sample_data(sample_config, sample_paper, sample_summary):
    """Test paper classification using sample data."""
    agent = ClassifierAgent(sample_config, use_sample_data=True)
    result = agent.classify_paper(sample_paper, sample_summary)
    
    assert isinstance(result, dict)
    assert "category" in result
    assert "confidence" in result
    assert "rationale" in result
    assert result["category"] == "Large Language Models"
    assert 0 <= result["confidence"] <= 1

@patch('openai.OpenAI')
def test_classify_paper_with_api(mock_openai, sample_config, sample_paper, 
                               sample_summary, mock_openai_response):
    """Test paper classification using the OpenAI API."""
    # Setup mock
    mock_client = Mock()
    mock_client.chat.completions.create.return_value = mock_openai_response
    mock_openai.return_value = mock_client
    
    agent = ClassifierAgent(sample_config)
    result = agent.classify_paper(sample_paper, sample_summary)
    
    assert result["category"] == "Large Language Models"
    assert result["confidence"] == 0.9
    assert "rationale" in result
    mock_client.chat.completions.create.assert_called_once()

@patch('openai.OpenAI')
def test_classify_paper_api_error(mock_openai, sample_config, sample_paper, sample_summary):
    """Test error handling when API call fails."""
    # Setup mock to raise an exception
    mock_client = Mock()
    mock_client.chat.completions.create.side_effect = Exception("API Error")
    mock_openai.return_value = mock_client
    
    agent = ClassifierAgent(sample_config)
    
    with pytest.raises(RuntimeError) as exc_info:
        agent.classify_paper(sample_paper, sample_summary)
    assert "Failed to classify paper" in str(exc_info.value)

def test_build_classification_prompt(sample_config, sample_paper, sample_summary):
    """Test prompt building functionality."""
    agent = ClassifierAgent(sample_config)
    prompt = agent._build_classification_prompt(sample_paper, sample_summary)
    
    assert sample_paper["title"] in prompt
    assert sample_paper["summary"] in prompt
    assert sample_summary in prompt
    assert "Large Language Models" in prompt
    assert "Computer Vision" in prompt
    assert "classification" in prompt.lower()
    assert "json" in prompt.lower()

def test_parse_classification_result_valid_json(sample_config):
    """Test parsing of valid JSON classification result."""
    agent = ClassifierAgent(sample_config)
    test_result = """
    ```json
    {
        "category": "Large Language Models",
        "confidence": 0.9,
        "rationale": "Test rationale"
    }
    ```
    """
    
    result = agent._parse_classification_result(test_result)
    assert result["category"] == "Large Language Models"
    assert result["confidence"] == 0.9
    assert result["rationale"] == "Test rationale"

def test_parse_classification_result_invalid_json(sample_config):
    """Test parsing of invalid JSON classification result."""
    agent = ClassifierAgent(sample_config)
    test_result = "Invalid JSON content"
    
    with pytest.raises(ValueError) as exc_info:
        agent._parse_classification_result(test_result)
    assert "Invalid JSON format" in str(exc_info.value)

def test_parse_classification_result_missing_fields(sample_config):
    """Test parsing of JSON result with missing required fields."""
    agent = ClassifierAgent(sample_config)
    test_result = """
    ```json
    {
        "category": "Large Language Models"
    }
    ```
    """
    
    with pytest.raises(ValueError) as exc_info:
        agent._parse_classification_result(test_result)
    assert "Missing required fields" in str(exc_info.value)

def test_parse_classification_result_invalid_confidence(sample_config):
    """Test parsing of JSON result with invalid confidence value."""
    agent = ClassifierAgent(sample_config)
    test_result = """
    ```json
    {
        "category": "Large Language Models",
        "confidence": 1.5,
        "rationale": "Test rationale"
    }
    ```
    """
    
    with pytest.raises(ValueError) as exc_info:
        agent._parse_classification_result(test_result)
    assert "Invalid confidence value" in str(exc_info.value)

def test_parse_classification_result_unexpected_category(sample_config):
    """Test parsing of JSON result with unexpected category."""
    agent = ClassifierAgent(sample_config)
    test_result = """
    ```json
    {
        "category": "Unexpected Category",
        "confidence": 0.9,
        "rationale": "Test rationale"
    }
    ```
    """
    
    result = agent._parse_classification_result(test_result)
    assert result["category"] == "Unexpected Category"
    # The warning will be logged but the result is still returned 