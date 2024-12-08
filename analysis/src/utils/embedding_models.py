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
from torch_geometric.nn import Node2Vec as PyGNode2Vec
import analysis_constants 

TRANSR_NUM_EPOCHS = 1000
TRANSR_PRINT_EVERY = 100
TRANSR_LEARNING_RATE = 0.01
TRANSR_MARGIN = 1.0

NODE2VEC_NUM_EPOCHS =100
NODE2VEC_WALK_LENGTH=80
NODE2VEC_CONTEXT_SIZE=10
NODE2VEC_WALKS_PER_NODE=10 

class TransREmbedding(torch.nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim=analysis_constants.EMBEDDING_DIM):
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
        margin = TRANSR_MARGIN
        loss = torch.max(pos_distance - neg_distance + margin, torch.tensor(0.0))

        return loss

def train_embeddings_transr(graph_data, epochs=TRANSR_NUM_EPOCHS):

    model = TransREmbedding(
        num_entities=len(graph_data[analysis_constants.ENTITIES_KEY]),
        num_relations=len(graph_data[analysis_constants.RELATIONS_KEY])
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=TRANSR_LEARNING_RATE)

    for epoch in range(epochs):
        total_loss = 0

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
        if (epoch+1) % TRANSR_PRINT_EVERY == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss}")
    return model


def train_node2vec(graph_data, epochs=NODE2VEC_NUM_EPOCHS, embedding_dim=analysis_constants.EMBEDDING_DIM, 
                   walk_length=NODE2VEC_WALK_LENGTH, context_size=NODE2VEC_CONTEXT_SIZE, walks_per_node=NODE2VEC_WALKS_PER_NODE):
    # Create edge index for the graph
    edge_index = torch.tensor([
        [graph_data[analysis_constants.ENTITY_MAPPING_KEY][source], 
         graph_data[analysis_constants.ENTITY_MAPPING_KEY][target]]
        for source, _, target in graph_data[analysis_constants.TRIPLES_KEY]
    ], dtype=torch.long).t().contiguous()
    
    model = PyGNode2Vec(
        edge_index, 
        embedding_dim=embedding_dim, 
        walk_length=walk_length,
        context_size=context_size,
        walks_per_node=walks_per_node
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        pos_sample, neg_sample = model.sample()
        
        optimizer.zero_grad()
        loss = model.loss(pos_sample, neg_sample)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss}")


def train_embeddings(graph_data, model_name): 
    assert model_name in analysis_constants.EMBEDDING_MODELS, f"Model name {model_name} not supported"
    if model_name == analysis_constants.TRANSR: 
        return train_embeddings_transr(graph_data)
    
    else:
        raise ValueError(f"Model name {model_name} not supported")
    
