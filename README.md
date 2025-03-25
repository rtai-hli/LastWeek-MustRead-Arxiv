# LastWeek-MustRead-Arxiv

An intelligent arXiv paper analysis system that automatically discovers, processes, and ranks the most relevant recent research papers based on your interests.

## Features

- 🤖 Automated daily paper collection from arXiv
- 📊 Multi-agent system architecture for sophisticated paper analysis:
  - Paper scraping from arXiv
  - Intelligent summarization
  - Topic classification
  - Novelty assessment
  - Relevance scoring
- 🗃️ Persistent storage of paper analyses in SQLite database
- 📈 Daily report generation with ranked papers
- 🎯 Configurable interest areas and categories
- ⚙️ Flexible scheduling (run once or schedule daily)

## Requirements

- Python 3.x
- OpenAI API key
- Dependencies listed in `requirements.txt`

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Create a `.env` file with your OpenAI API key:
```bash
OPENAI_API_KEY=your_api_key_here
```

## Usage

Run once immediately:
```bash
python main.py --mode once
```

Run as a scheduled service (runs daily at 2 AM):
```bash
python main.py --mode schedule
```

## Configuration

Edit the `setup_config()` function in `main.py` to customize:
- Interested research fields
- arXiv categories to monitor
- Maximum papers to process per run
- LLM model and parameters
- Database path

## Project Structure

- `main.py`: Core application logic and pipeline orchestration
- `agents/`: Individual AI agents for different analysis tasks
- `database/`: Database management and persistence layer
- `papers.db`: SQLite database for storing paper analyses
- `requirements.txt`: Project dependencies

## Output

The system generates:
- Daily CSV reports with ranked papers
- Detailed paper analyses in the SQLite database
- Logging information about the analysis process

## License

MIT License

Copyright (c) 2025 @RTAI-HLI

This project is a fork of https://github.com/jyguo/AnAgent. While I've chosen the MIT License for this fork, I want to explicitly acknowledge and credit the original work.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
