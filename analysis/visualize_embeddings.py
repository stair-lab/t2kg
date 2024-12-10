import argparse
import os
import analysis_constants
import sys
sys.path.append(analysis_constants.UTILS_PATH)
import graph_utils
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib import colormaps
import numpy as np
from matplotlib.colors import ListedColormap

COLOR_MAP = {}
NODE_COLOR = 'gray'
LOW_RES = False 
if LOW_RES:
    FIGURE_SIZE = (15, 10)
    DPI = 100
    FONT_SIZE = 8
else:
    FIGURE_SIZE = (60, 40)
    DPI = 400
    FONT_SIZE = 3 

def plot_embeddings_for_graph(saved_graph_info_dir, plot_dir):
    entity_embeddings, _, graph_data = graph_utils.load_saved_graph_data(saved_graph_info_dir)
    entity_to_id_map = graph_data[analysis_constants.ENTITY_MAPPING_KEY]
    cluster_assignments = graph_data[analysis_constants.ENTITY_CLUSTERS_KEY]  # Adjust key name as per your data
    
    # Standardize and reduce embeddings to 2D
    scaler = StandardScaler()
    scaled_embeddings = scaler.fit_transform(entity_embeddings)

    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(scaled_embeddings)

    # Assign colors to clusters
    unique_clusters = np.unique(list(cluster_assignments))
    base_colormap = colormaps.get_cmap('tab10')  # Get the base colormap
    cluster_colors = ListedColormap(base_colormap(np.linspace(0, 1, len(unique_clusters))))
    color_map = {cluster: cluster_colors(i) for i, cluster in enumerate(unique_clusters)}

    # Plot nodes with cluster-based coloring
    plt.figure(figsize=FIGURE_SIZE)
    for entity, idx in entity_to_id_map.items():
        x, y = reduced_embeddings[idx]
        cluster = cluster_assignments[idx]  # Retrieve the cluster for the current entity
        color = color_map.get(cluster, NODE_COLOR)
        plt.scatter(x, y, c=[color], label=f"Cluster {cluster}" if idx == 0 else "", s=20)  # Avoid duplicate labels
        plt.annotate(entity, (x, y), xytext=(5, 5),
                     textcoords='offset points',
                     fontsize=FONT_SIZE,
                     color='black')

    plt.title('Entity Embeddings with Cluster Coloring')
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
        "-d",
        "--directory",
        type=str,
        default=analysis_constants.EMBEDDINGS_BASE_PATH,  # Default placeholder value
        help="Path to the directory containing saved data for graphs."
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=analysis_constants.EMBEDDING_PLOTS_BASE_PATH,  # Default placeholder value
        help="Path to save the visualizations."
    )

    args = parser.parse_args()
    visualize_embeddings(None, directory=args.directory, output=args.output)