import sys
import os
from analysis_scripts.utils import graph_utils

# Get the directory of the current script file
script_dir = os.path.dirname(os.path.abspath(__file__))
kg_path = os.path.join(script_dir, "../extracted_graphs/test_small.json")


def run_kg_graph_analysis():
    data = graph_utils.load_kg_data(kg_path)
    print(data)


if __name__ == '__main__':
    run_kg_graph_analysis()