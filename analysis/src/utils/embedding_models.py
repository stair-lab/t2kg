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
from torch.utils.data import DataLoader, Dataset
import analysis_constants 

TRANSR_NUM_EPOCHS = 5000
TRANSR_PRINT_EVERY = 100
TRANSR_LEARNING_RATE = 0.01
TRANSR_MARGIN = 1.0
TRANSR_BATCH_SIZE = 32

NODE2VEC_NUM_EPOCHS =100
NODE2VEC_WALK_LENGTH=80
NODE2VEC_CONTEXT_SIZE=10
NODE2VEC_WALKS_PER_NODE=10 


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class KnowledgeGraphDataset(Dataset):
    def __init__(self, relations, entity_mapping, relation_mapping):
        self.relations = relations
        self.entity_mapping = entity_mapping
        self.relation_mapping = relation_mapping

    def __len__(self):
        return len(self.relations)

    def __getitem__(self, idx):
        source, pred, target = self.relations[idx]
        source_idx = self.entity_mapping[source]
        target_idx = self.entity_mapping[target]
        relation_idx = self.relation_mapping[pred]
        return source_idx, relation_idx, target_idx


class TransREmbedding(torch.nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim=analysis_constants.EMBEDDING_DIM):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.llm_model = AutoModel.from_pretrained("bert-base-uncased").to(device)
        self.entity_embeddings = torch.nn.Embedding(num_entities, embedding_dim).to(device)
        self.relation_embeddings = torch.nn.Embedding(num_relations, embedding_dim).to(device)

        self.relation_matrices = torch.nn.ParameterList([
            torch.nn.Parameter(torch.randn(embedding_dim, embedding_dim).to(device))
            for _ in range(num_relations)
        ])

    def get_llm_embeddings(self, entities):
        embeddings = []
        for entity in entities:
            inputs = self.tokenizer(entity, return_tensors="pt", padding=True, truncation=True).to(device)
            with torch.no_grad():
                outputs = self.llm_model(**inputs)
                embeddings.append(outputs.last_hidden_state.mean(dim=1))
        return torch.cat(embeddings)

    def forward(self, edge_index, edge_type, entities):
        x = self.get_llm_embeddings(entities)
        x = x + self.entity_embeddings(torch.arange(len(entities)).to(device))
        return x

    def link_prediction_loss(self, pos_edges, neg_edges, relation_types):
        pos_head, pos_tail = pos_edges
        neg_head, neg_tail = neg_edges

        pos_distance = torch.norm(
            pos_head + self.relation_embeddings(relation_types) - pos_tail, dim=1
        )
        neg_distance = torch.norm(
            neg_head + self.relation_embeddings(relation_types) - neg_tail, dim=1
        )
        margin = TRANSR_MARGIN
        loss = torch.mean(torch.relu(pos_distance - neg_distance + margin))
        return loss


def train_embeddings_transr(graph_data, epochs=TRANSR_NUM_EPOCHS):
    model = TransREmbedding(
        num_entities=len(graph_data[analysis_constants.ENTITIES_KEY]),
        num_relations=len(graph_data[analysis_constants.RELATIONS_KEY])
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=TRANSR_LEARNING_RATE)

    entities = graph_data[analysis_constants.ENTITIES_KEY]
    entity_mapping = graph_data[analysis_constants.ENTITY_MAPPING_KEY]
    relation_mapping = graph_data[analysis_constants.RELATION_MAPPING_KEY]
    triples = graph_data[analysis_constants.TRIPLES_KEY]

    dataset = KnowledgeGraphDataset(
        relations=triples,
        entity_mapping=entity_mapping,
        relation_mapping=relation_mapping
    )
    data_loader = DataLoader(dataset, batch_size=TRANSR_BATCH_SIZE, shuffle=True)

    for epoch in range(epochs):
        total_loss = 0
        for batch in data_loader:
            source_batch, relation_batch, target_batch = [b.to(device) for b in batch]

            neg_source_batch = torch.randint(0, len(entities), source_batch.size(), device=device)
            neg_target_batch = torch.randint(0, len(entities), target_batch.size(), device=device)

            pos_heads = model.entity_embeddings(source_batch)
            pos_tails = model.entity_embeddings(target_batch)
            neg_heads = model.entity_embeddings(neg_source_batch)
            neg_tails = model.entity_embeddings(neg_target_batch)

            loss = model.link_prediction_loss(
                (pos_heads, pos_tails),
                (neg_heads, neg_tails),
                relation_batch
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/ {epochs}, Loss: {total_loss}")

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
    
