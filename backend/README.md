# Reddit Analysis Backend

This is the backend component of the Reddit Analysis application. It provides functionality for analyzing Reddit content, including topic extraction, toxicity analysis, and positive content identification.

## Directory Structure

```
backend/
├── src/                    # Source code
│   ├── analysis/          # Analysis-related modules
│   ├── data_fetchers/     # Data fetching modules
│   ├── database/          # Database-related modules
│   ├── utils/             # Utility functions
│   └── main.py            # Main application entry point
├── config/                # Configuration files
├── data/                  # Data files
├── tests/                 # Test files
├── requirements/          # Requirements files
└── docs/                  # Documentation
```

## Components

- **analysis**: Contains modules for NLP analysis, word embeddings, and visualization
- **data_fetchers**: Modules for fetching data from Reddit and other sources
- **database**: Database interaction and caching functionality
- **utils**: Utility functions and helper classes

## Setup

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements/requirements.txt
   ```

3. Copy the example environment file and configure it:
   ```bash
   cp config/.env.example config/.env
   ```

4. Run the application:
   ```bash
   python src/main.py
   ```

## Testing

To run tests, execute `pytest` in the terminal.

## Documentation

Additional documentation can be found in the `docs/` directory. 