import json
import networkx as nx
import matplotlib.pyplot as plt
from typing import List, Tuple
import json
from typing import List, Tuple
import networkx as nx
import matplotlib.pyplot as plt
from pydantic import BaseModel
from index import SyntheticData, GoldLabelDataset

def load_gold_label_dataset(file_path: str) -> List[GoldLabelDataset]:
    with open(file_path, 'r') as file:
        return [GoldLabelDataset(**json.loads(line)) for line in file]

def create_knowledge_graph(entities: List[str], relations: List[Tuple[str, str, str]]) -> nx.DiGraph:
    G = nx.DiGraph()
    G.add_nodes_from(entities)
    G.add_edges_from([(s, o, {'label': p}) for s, p, o in relations])
    return G

def visualize_knowledge_graph(G: nx.DiGraph, title: str) -> None:
    plt.figure(figsize=(16, 12))  # Increased figure size for better visibility
    pos = nx.spring_layout(G, k=1.5, iterations=100)  # Increased k and iterations for more spread
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=4000)
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, arrowsize=20)
    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
    
    plt.title(title, fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    
    # Add some padding around the graph
    x_values, y_values = zip(*pos.values())
    x_margin = (max(x_values) - min(x_values)) * 0.1
    y_margin = (max(y_values) - min(y_values)) * 0.1
    plt.xlim(min(x_values) - x_margin, max(x_values) + x_margin)
    plt.ylim(min(y_values) - y_margin, max(y_values) + y_margin)
    
    plt.savefig(f"./extracted/{title.replace(' ', '_')}.png", dpi=300, bbox_inches='tight')
    plt.close()

def process_sample(sample: GoldLabelDataset) -> None:
    synthetic_data = sample.synthetic
    
    # Chosen Knowledge Graph
    chosen_G = create_knowledge_graph(synthetic_data.chosen_entities_all, synthetic_data.chosen_relations)
    visualize_knowledge_graph(chosen_G, f"Chosen KG - {sample.user_query[:20]}...")
    
    # Rejected Knowledge Graph
    rejected_G = create_knowledge_graph(synthetic_data.rejected_entities_all, synthetic_data.rejected_relations)
    visualize_knowledge_graph(rejected_G, f"Rejected KG - {sample.user_query[:20]}...")

def main():
    dataset = load_gold_label_dataset('./extracted/gold_label_dataset.jsonl')
    for sample in dataset:
        try:
            process_sample(sample)
        except Exception as e:
            print(f"Error processing sample: {e}")

if __name__ == "__main__":
    main()
