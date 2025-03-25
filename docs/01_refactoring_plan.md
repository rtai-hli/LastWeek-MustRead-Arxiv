# Refactoring Plan

## 1. ArXiv Integration Refactoring

### Current Status
- Project currently uses a custom scraping agent
- ArXiv has an official Python library (`arxiv==2.1.3`) already in requirements.txt

### Planned Changes
1. Remove `ArxivScraperAgent` class
2. Create new `ArxivFetcher` class using official API
   - Implement paper fetching using `arxiv` library
   - Add proper error handling and rate limiting
   - Include metadata extraction improvements
3. Update main pipeline to use new fetcher
4. Add caching layer to prevent redundant API calls

### Migration Steps
1. [ ] Create new `arxiv_fetcher.py` module
2. [ ] Implement and test new fetcher with arxiv library
3. [ ] Update main.py to use new fetcher
4. [ ] Remove old scraper agent
5. [ ] Add tests for new fetcher

## 2. Code Structure Improvements

### Rename and Reorganize
1. [ ] Rename `demo.py` to `arxiv_weekly.py`
2. [ ] Create proper package structure:
```
lastweek_mustread_arxiv/
├── src/
│   ├── __init__.py
│   ├── arxiv_weekly.py
│   ├── fetchers/
│   ├── agents/
│   ├── database/
│   └── utils/
├── tests/
├── docs/
└── examples/
```

### Testing Infrastructure
1. [ ] Set up pytest framework
2. [ ] Add unit tests for each component
3. [ ] Set up GitHub Actions for CI/CD
4. [ ] Add integration tests

### Environment Management
1. [ ] Create separate requirements for dev, test, and prod
2. [ ] Add proper logging configuration
3. [ ] Implement configuration management 