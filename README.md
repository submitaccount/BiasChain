# BiasChain
# Peersum Bias Detection
## Overview
Peersum Bias Detection is a Python project that analyzes peer review texts for bias, consistency, sentiment, and more. It leverages state-of-the-art language model integrations and schema-based output parsing with tools like [langchain_google_genai](https://github.com/langchain-ai/langchain).

## Project Structure
```
.env
.gitignore
LICENSE
README.md
requirements.txt
data/
    prepare_sample_subset.py         # Data preparation scripts
    split_dataset.py
    processed/
        bias_detection/
        consistency_check/
        pairwise_comparison/
        review_extraction/
        sentiment_analysis/
src/
    bias_detection/                  # Bias detection logic and prompts
        bias_detection_tools.py
        prompts.py
        revised_bias_detection_tools.py
    consistency_check/               # Consistency check implementation
        consistency_checker_chain.py
    pairwise_comparison/             # Review comparison functionality
        ... (additional files)
    reviews_extraction/              # Review extraction implementation
    sentiment_analysis/              # Sentiment analysis tools and chains
```

## Key Components

- **Bias Detection:**  
  Uses custom prompts and schema-based parsers to detect topic and methodological bias in reviews. See [`src/bias_detection/prompts.py`](src/bias_detection/prompts.py).

- **Consistency Check:**  
  Validates the internal consistency of reviews via chain-of-thought analysis. See [`src/consistency_check/consistency_checker_chain.py`](src/consistency_check/consistency_checker_chain.py).

- **Pairwise Comparison:**  
  Compares individual reviews against a set of reviews to reveal alignment and contradictions. See [`src/pairwise_comparison/pairwise_comparison_chain.py`](src/pairwise_comparison/pairwise_comparison_chain.py).

- **Sentiment Analysis:**  
  Evaluates the tone and sentiment of peer reviews. See [`src/sentiment_analysis/sentiment_analyzer_chain.py`](src/sentiment_analysis/sentiment_analyzer_chain.py) and [`src/sentiment_analysis/structured_tool.py`](src/sentiment_analysis/structured_tool.py).

## Setup

1. **Install Dependencies:**  
   Ensure you have Python installed, then run:
   ```sh
   pip install -r requirements.txt
   ```

2. **Configure Environment Variables:**  
   Create a `.env` file in the root directory and define necessary variables such as `GEMINI_API_KEY`.

## Running the Project

- **Data Preparation:**  
  Use scripts inside the `data/` directory (e.g., `prepare_sample_subset.py`, `split_dataset.py`) to set up your dataset.

- **Analyze Reviews:**  
  Run the appropriate analysis chain in `src/` (e.g., for bias detection, consistency checks, pairwise comparisons, or sentiment analysis).

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
