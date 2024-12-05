# Extract Knowledge Graphs

This directory contains scripts for extracting knowledge graphs from text data.
- `index.py`: Extract entities and relations from text using GPT-4o.
- `aggregate.py`: Aggregate data from multiple JSON Lines (jsonl) files into a single dataset.
- `cluster.py`: Cluster entities using GPT-4o and create an entity enumeration.
- `visualize.py`: Visualize knowledge graphs using NetworkX and Matplotlib.

## Run

You'll need to set the `OPENAI_API_KEY` environment variable to your OpenAI API key before running the scripts.
Also, be sure to change the file path variables in each script to make sure they point to the correct input and output paths. (sorry for the inconvenience)

Run in this order:
- `python index.py`
- `python aggregate.py`
- `python cluster.py`
- `python visualize.py`
