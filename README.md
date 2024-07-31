# simple-llm-endpoint-benchmark-tool

This is a simple tool to benchmark the performance of a LLM endpoint. It is written in Python gives you the ability to test the performance of a LLM endpoint by sending multiple requests to the endpoint and measuring the time it takes to get a response.

## Installation

To install the tool, you need to have Python 3 installed on your machine. You can download Python 3 from the official website: https://www.python.org/downloads/

Once you have Python 3 installed, you can install the tool by running the following command:

```bash
pip install -r requirements.txt
```

This will install all the required dependencies for the tool.

## Usage

Rename the .env.example file to .env and update the values with your own values.

```bash
OPENAI_ENDPOINT="http://localhost:11434"
OPENAI_API_KEY="ollama"
OPENAI_MAX_TOKENS=512
OPENAI_MODEL="mistral"
IS_INSTRUCT_MODEL="True"
DATASET="oasst1"
PARALLEL_CONVERSATIONS=1
MESSAGE_COUNT=3
MIN_CONTEXT_MESSAGES=3
MAX_CONTEXT_MESSAGES=7
```

You can then run the tool by running the following command:

```bash
python src/main.py
```

This will start the benchmarking process and output the results to the console.
It will also save the results to a Excel file in the `benchmark-results` directory.