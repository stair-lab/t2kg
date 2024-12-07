import torch
from torch_geometric.nn import RGCNConv
from torch_geometric.data import Data
import torch.nn.functional as F
import numpy as np
from transformers import AutoModel, AutoTokenizer
from torch_geometric.nn import GraphConv
from typing import Dict, List
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from torch_geometric.nn import GCNConv
from pykeen.models import TransE
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory

import analysis_constants 

# Define the RGCN model
class RGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_relations):
        super(RGCN, self).__init__()
        self.rgcn1 = RGCNConv(in_channels, hidden_channels, num_relations)
        self.rgcn2 = RGCNConv(hidden_channels, out_channels, num_relations)

    def forward(self, x, edge_index, edge_type):
        x = self.rgcn1(x, edge_index, edge_type)
        x = torch.relu(x)
        x = self.rgcn2(x, edge_index, edge_type)
        return x
    

# adapted from colab 2 of the homework 
class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout, return_embeds=False):
        super(GCN, self).__init__()
        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList(
            [GCNConv(input_dim if i == 0 else hidden_dim, output_dim if i == num_layers - 1 else hidden_dim) 
             for i in range(num_layers)]
        )
        self.bns = torch.nn.ModuleList([torch.nn.BatchNorm1d(hidden_dim) for _ in range(num_layers - 1)])
        self.softmax = torch.nn.LogSoftmax()
        self.dropout = dropout
        self.return_embeds = return_embeds

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t):
        H = x
        for i in range(self.num_layers - 1):
            H = self.convs[i](H, adj_t)
            H = self.bns[i](H)
            H = torch.nn.functional.relu(H)
            H = torch.nn.functional.dropout(H, p=self.dropout, training=self.training)
        H = self.convs[self.num_layers - 1](H, adj_t)
        return H if self.return_embeds else self.softmax(H)
    
class TransREmbedding(torch.nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim=64):
        super().__init__()
        # LLM-based entity embeddings
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.llm_model = AutoModel.from_pretrained("bert-base-uncased")
        # Trainable embeddings
        self.entity_embeddings = torch.nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = torch.nn.Embedding(num_relations, embedding_dim)
        # Relation-specific projection matrices
        self.relation_matrices = torch.nn.ParameterList([
            torch.nn.Parameter(torch.randn(embedding_dim, embedding_dim))
            for _ in range(num_relations)
        ])
    def get_llm_embeddings(self, entities: List[str]) -> torch.Tensor:
        # Use LLM to get initial embeddings
        embeddings = []
        for entity in entities:
            inputs = self.tokenizer(entity, return_tensors="pt", padding=True, truncation=True)
            with torch.no_grad():
                outputs = self.llm_model(**inputs)
                embeddings.append(outputs.last_hidden_state.mean(dim=1))
        return torch.stack(embeddings)
    def forward(self, edge_index, edge_type, entities):
        # Initial LLM embeddings
        x = self.get_llm_embeddings(entities)
        # Combine with trainable embeddings
        x = x + self.entity_embeddings(torch.arange(len(entities)))
        return x
    
    
    def link_prediction_loss(self, pos_edges, neg_edges, relation_type):
        # TransR-style margin-based loss
        pos_head, pos_tail = pos_edges
        neg_head, neg_tail = neg_edges

        proj_matrix = self.relation_matrices[relation_type]

        pos_head_proj = torch.matmul(pos_head, proj_matrix)
        pos_tail_proj = torch.matmul(pos_tail, proj_matrix)
        neg_head_proj = torch.matmul(neg_head, proj_matrix)
        neg_tail_proj = torch.matmul(neg_tail, proj_matrix)

        pos_distance = torch.norm(pos_head_proj + self.relation_embeddings(torch.tensor(relation_type)) - pos_tail_proj)
        neg_distance = torch.norm(neg_head_proj + self.relation_embeddings(torch.tensor(relation_type)) - neg_tail_proj)

        # Margin-based ranking loss
        margin = 1.0
        loss = torch.max(pos_distance - neg_distance + margin, torch.tensor(0.0))

        return loss

def train_embeddings_transr(graph_data, epochs=10):

    # Initialize model
    model = TransREmbedding(
        num_entities=len(graph_data['entities']),
        num_relations=len(graph_data['relations'])
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Training loop
    for epoch in range(epochs):
        total_loss = 0

        # Sample positive and negative edges
        for relation in graph_data[analysis_constants.TRIPLES_KEY]:      
            source, pred, target = relation
            # Get entity indices
            source_idx = graph_data[analysis_constants.ENTITY_MAPPING_KEY][source]
            target_idx = graph_data[analysis_constants.ENTITY_MAPPING_KEY][target]
            relation_idx = graph_data[analysis_constants.RELATION_MAPPING_KEY][pred]

            # Create positive and negative samples
            pos_edges = (
                model.entity_embeddings(torch.tensor([source_idx])),
                model.entity_embeddings(torch.tensor([target_idx]))
            )
            # Generate negative samples (random entities)
            neg_source_idx = torch.randint(0, len(graph_data['entities']), (1,))
            neg_target_idx = torch.randint(0, len(graph_data['entities']), (1,))

            neg_edges = (
                model.entity_embeddings(neg_source_idx),
                model.entity_embeddings(neg_target_idx)
            )
            # Compute loss
            loss = model.link_prediction_loss(pos_edges, neg_edges, relation_idx)

            # Optimization step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss}")
    return model


def train_embeddings(graph_data, model_name): 
    assert model_name in analysis_constants.EMBEDDING_MODELS, f"Model name {model_name} not supported"
    if model_name == analysis_constants.TRANSR: 
        return train_embeddings_transr(graph_data)
    
    elif model_name == analysis_constants.RGCN: 
        raise NotImplementedError
    
    elif model_name == analysis_constants.GCN: 
        raise NotImplementedError
    
