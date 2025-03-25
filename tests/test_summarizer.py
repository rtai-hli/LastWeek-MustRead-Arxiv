"""Tests for the SummarizerAgent class."""

import pytest
from unittest.mock import Mock, patch
from src.agents.summarizer import SummarizerAgent

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
def mock_openai_response():
    """Create a mock OpenAI API response."""
    mock_response = Mock()
    mock_response.choices = [
        Mock(
            message=Mock(
                content="""
                Research Problem: Test research problem
                Methodology: Test methodology
                Key Innovations: Test innovations
                Findings/Results: Test findings
                Potential Impact: Test impact
                """
            )
        )
    ]
    return mock_response

def test_summarizer_initialization(sample_config):
    """Test SummarizerAgent initialization."""
    agent = SummarizerAgent(sample_config)
    assert agent.model == "gpt-4-turbo-preview"
    assert agent.temperature == 0.7
    assert "expert AI paper summarization specialist" in agent.system_message

def test_summarizer_initialization_with_sample_data(sample_config):
    """Test SummarizerAgent initialization with sample data mode."""
    agent = SummarizerAgent(sample_config, use_sample_data=True)
    assert agent.use_sample_data is True
    assert not hasattr(agent, 'client')

@patch('openai.OpenAI')
def test_summarize_paper_with_sample_data(mock_openai, sample_config, sample_paper):
    """Test paper summarization using sample data."""
    agent = SummarizerAgent(sample_config, use_sample_data=True)
    summary = agent.summarize_paper(sample_paper)
    
    assert isinstance(summary, dict)
    assert "research_problem" in summary
    assert "methodology" in summary
    assert "innovations" in summary
    assert "findings" in summary
    assert "impact" in summary
    mock_openai.assert_not_called()

@patch('openai.OpenAI')
def test_summarize_paper_with_api(mock_openai, sample_config, sample_paper, mock_openai_response):
    """Test paper summarization using the OpenAI API."""
    # Setup mock
    mock_client = Mock()
    mock_client.chat.completions.create.return_value = mock_openai_response
    mock_openai.return_value = mock_client
    
    agent = SummarizerAgent(sample_config)
    summary = agent.summarize_paper(sample_paper)
    
    assert isinstance(summary, dict)
    assert all(key in summary for key in ["research_problem", "methodology", "innovations", "findings", "impact"])
    mock_client.chat.completions.create.assert_called_once()

@patch('openai.OpenAI')
def test_summarize_paper_api_error(mock_openai, sample_config, sample_paper):
    """Test error handling when API call fails."""
    # Setup mock to raise an exception
    mock_client = Mock()
    mock_client.chat.completions.create.side_effect = Exception("API Error")
    mock_openai.return_value = mock_client
    
    agent = SummarizerAgent(sample_config)
    
    with pytest.raises(RuntimeError) as exc_info:
        agent.summarize_paper(sample_paper)
    assert "Failed to generate summary" in str(exc_info.value)

def test_build_summarization_prompt(sample_config, sample_paper):
    """Test prompt building functionality."""
    agent = SummarizerAgent(sample_config)
    prompt = agent._build_summarization_prompt(sample_paper)
    
    assert sample_paper["title"] in prompt
    assert "Author One" in prompt
    assert "Author Two" in prompt
    assert sample_paper["summary"] in prompt
    assert sample_paper["text_content"] in prompt
    assert "Research Problem:" in prompt
    assert "Methodology:" in prompt
    assert "Key Innovations:" in prompt

def test_parse_summary_sections(sample_config):
    """Test parsing of summary sections."""
    agent = SummarizerAgent(sample_config)
    test_summary = """
    Research Problem: Test problem
    Methodology: Test method
    Key Innovations: Test innovation
    Findings/Results: Test findings
    Potential Impact: Test impact
    """
    
    sections = agent._parse_summary_sections(test_summary)
    
    assert sections["research_problem"] == "Test problem"
    assert sections["methodology"] == "Test method"
    assert sections["innovations"] == "Test innovation"
    assert sections["findings"] == "Test findings"
    assert sections["impact"] == "Test impact"

def test_parse_summary_sections_with_alternative_headers(sample_config):
    """Test parsing of summary sections with alternative header formats."""
    agent = SummarizerAgent(sample_config)
    test_summary = """
    Research Problem: Test problem
    Main Methods: Test method
    Core Innovations: Test innovation
    Findings: Test findings
    Impact: Test impact
    """
    
    sections = agent._parse_summary_sections(test_summary)
    
    assert sections["research_problem"] == "Test problem"
    assert sections["methodology"] == "Test method"
    assert sections["innovations"] == "Test innovation"
    assert sections["findings"] == "Test findings"
    assert sections["impact"] == "Test impact"

def test_parse_summary_sections_with_missing_sections(sample_config):
    """Test parsing of summary sections with missing sections."""
    agent = SummarizerAgent(sample_config)
    test_summary = """
    Research Problem: Test problem
    Findings: Test findings
    """
    
    sections = agent._parse_summary_sections(test_summary)
    
    assert sections["research_problem"] == "Test problem"
    assert sections["methodology"] == ""
    assert sections["innovations"] == ""
    assert sections["findings"] == "Test findings"
    assert sections["impact"] == "" 