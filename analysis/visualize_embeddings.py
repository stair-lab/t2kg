import argparse
import os
import analysis_constants
import sys
sys.path.append(analysis_constants.UTILS_PATH)
import graph_utils
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

COLOR_MAP = {}
NODE_COLOR = 'gray'
FIGURE_SIZE = (15, 10)
DPI = 400

def plot_embeddings_for_graph(saved_graph_info_dir, plot_dir):
    entity_to_id_map, entity_embeddings = graph_utils.load_plotting_data(saved_graph_info_dir)

    scaler = StandardScaler()
    scaled_embeddings = scaler.fit_transform(entity_embeddings)

    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(scaled_embeddings)

    plt.figure(figsize=FIGURE_SIZE)
    color_map = COLOR_MAP

    for entity, idx in entity_to_id_map.items():
        x, y = reduced_embeddings[idx]
        color = color_map.get(entity, NODE_COLOR)
        plt.scatter(x, y, c=color)
        plt.annotate(entity, (x, y), xytext=(5, 5),
                     textcoords='offset points',
                     fontsize=8,
                     color='black')

    plt.title('Entity Embeddings after Training')
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.tight_layout()

    print("PCA Explained Variance:")
    print(f"First component: {pca.explained_variance_ratio_[0]*100:.2f}%")
    print(f"Second component: {pca.explained_variance_ratio_[1]*100:.2f}%")

    os.makedirs(plot_dir, exist_ok=True)  
    plot_path = graph_utils.get_plot_image_path(saved_graph_info_dir, plot_dir)
    plt.savefig(plot_path, format='png', dpi=DPI)
    print(f"Plot saved to {plot_path}")

def visualize_embeddings(selected, directory, output):
    if selected is None:
        selected = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]

    for graph_name in selected:
        print(f"Visualizing: {graph_name}")
        saved_graph_info_dir = os.path.join(directory, graph_name)
        plot_embeddings_for_graph(saved_graph_info_dir, output)

        # Add more visualizations if possible here 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize embeddings for knowledge graphs.")
    parser.add_argument(
        "-s",
        "--selected",
        type=str,
        nargs="+",
        default=None,
        help="Names of the graphs to visualize. Defaults to all graphs in the directory."
    )
    parser.add_argument(
        "-d",
        "--directory",
        type=str,
        default=analysis_constants.EMBEDDINGS_BASE_PATH,  # Default placeholder value
        help="Path to the directory containing saved data for graphs. Default is './data'."
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=analysis_constants.EMBEDDING_PLOTS_BASE_PATH,  # Default placeholder value
        help="Path to save the visualizations."
    )

    args = parser.parse_args()
    visualize_embeddings(selected=args.selected, directory=args.directory, output=args.output)