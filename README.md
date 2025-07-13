# AutoSummarizer CLI

**AutoSummarizer CLI** is a command-line tool that generates concise summaries of input text using modern NLP techniques and transformer-based models. Designed for ease of use, it supports both extractive and abstractive summarization with CPU-friendly performance.

## Features

- Command-line interface built with `click`
- Summarization using models from Hugging Face Transformers (`T5`, `BART`, etc.)
- Text preprocessing with `spaCy` and `nltk`
- Configurable summarization length and method
- Clean output with optional formatting (`rich`)
- Easily extendable for other NLP tasks

---

## Installation

1. **Clone the repository**
```bash
git clone https://github.com/UnboundSB/AutoSummarizer-CLI.git
cd AutoSummarizer-CLI
