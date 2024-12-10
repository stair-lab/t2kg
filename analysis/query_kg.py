import argparse
import os
import json
import analysis_constants
import sys
sys.path.append(analysis_constants.UTILS_PATH)
import graph_utils
import query_models

class KnowledgeGraphQuerySystem:
    def __init__(self, saved_graph_info_dir, input_txt_path, model_name):
        print(f"Loading query model: {model_name}")
        self.query_model = query_models.get_query_model(model_name, saved_graph_info_dir)
        self.query_results = {}
        if input_txt_path:
            with open(input_txt_path, 'r') as f:
                self.queries = [line.strip() for line in f if line.strip()]

            
    def execute_query(self, query):
        return self.query_model.predict(query)

    def query_from_file(self):
        for query in self.queries:
            result = self.execute_query(query)
            self.query_results[query] = result
      
    def interactive_query(self):
        print("Interactive Query Mode: Type 'exit' to quit.")
        while True:
            query = input("Enter your query: ")
            if query.lower() == analysis_constants.QUERY_SYSTEM_EXIT_COMMAND:
                print("Exiting interactive mode.")
                break
            result = self.execute_query(query)
            print("Query Results:", json.dumps(result, indent=4))
            self.query_results[query] = result

def run_batch_query_system(input_txt_path, output_dir, saved_graphs_info_dir): 
    for graph_name in os.listdir(saved_graphs_info_dir):
        saved_graph_info_dir = os.path.join(saved_graphs_info_dir, graph_name)
        run_query_system(input_txt_path, output_dir, saved_graph_info_dir)

def run_query_system(input_txt_path, output_dir, saved_graph_info_dir):
    query_system = KnowledgeGraphQuerySystem(
        saved_graph_info_dir, 
        input_txt_path, 
        args.model
    )

    query_system.query_from_file()
    query_results = query_system.query_results

    os.makedirs(output_dir, exist_ok=True)
    # saving the query reults
    output_file_path = graph_utils.get_query_results_path(saved_graph_info_dir, output_dir)
    print(f"Saving query results to: {output_file_path}")
    with open(output_file_path, 'w') as f:
        json.dump(query_results, f, indent=4)
    print("Query results saved successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query a knowledge graph.")
    parser.add_argument(
        "-f",
        "--file",
        type=str,
        required=True,
        help="Path to the file containing queries."
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=analysis_constants.QUERY_BASE_PATH,
        help=f"Directory to save the output for batch mode. Default is {analysis_constants.QUERY_BASE_PATH}"
    )

    parser.add_argument(
        "-d",
        "--directory",
        type=str,
        default=analysis_constants.EMBEDDINGS_BASE_PATH,
        help="Directory containing the directories of saved data of all kgs."
    )

    parser.add_argument(
        "-m",
        "--model",
        type=str,
        choices=["pinecone"],
        default=analysis_constants.DEFAULT_QUERY_MODEL,
        help="Model to use for querying the knowledge graph ('pinecone')."
    )

    args = parser.parse_args()

    print("Running query system...")
    run_batch_query_system(args.file, args.output, args.directory)
    
   

    
