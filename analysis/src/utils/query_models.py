import pinecone
import os
from pinecone import Pinecone, ServerlessSpec
import sys
import analysis_constants
sys.path.append(analysis_constants.UTILS_PATH)
import graph_utils
import torch
from transformers import AutoTokenizer, AutoModel

pc = Pinecone(analysis_constants.PINE_API_KEY)
SIMILARY_METRIC = 'cosine'
CLOUD = 'aws'
REGION = 'us-east-1'
BATCH_SIZE = 100
QUERY_TOP_K = 5
INCLUDE_PINCONE_METADATA = False

def create_kg_index( entity_embeddings, entities, _, entity_mapping, index_name):   
    if index_name not in pc.list_indexes().names():
        pc.create_index( name= index_name, dimension=analysis_constants.EMBEDDING_DIM, metric= SIMILARY_METRIC,
            spec=ServerlessSpec(
              cloud=CLOUD,
              region=REGION
            )
        )

    # Connect to index
    index = pc.Index(index_name)

    for i in range(0, len(entities), BATCH_SIZE):
        batch = entities[i:i+BATCH_SIZE]

        # Prepare vectors for upsert
        vectors = []
        for entity in batch:
            entity_idx = entity_mapping[entity]
            entity_emb = entity_embeddings[entity_idx]

            vectors.append((
                entity,  # Use entity name as ID
                entity_emb.tolist(),  # Embedding as list
                {
                    'type': 'entity',
                    'name': entity
                }  # Optional metadata
            ))
        index.upsert(vectors)

    return index

class PineconeQueryModel:
    def __init__(self, saved_graph_info_dir):
        index_name = graph_utils.get_index_name(saved_graph_info_dir)
        entity_embeddings, _, graph_data = graph_utils.load_saved_graph_data(saved_graph_info_dir)    

        entities = graph_data[analysis_constants.ENTITIES_KEY]
        relations = graph_data[analysis_constants.RELATIONS_KEY]    

        entity_to_idx = graph_data[analysis_constants.ENTITY_MAPPING_KEY]
        _ = graph_data[analysis_constants.RELATION_MAPPING_KEY]

        self.index = create_kg_index(
            entity_embeddings,
            entities,
            relations,
            entity_to_idx,
            index_name
        )
        self.entity_embeddings = entity_embeddings
        self.relations = relations
        self.entity_to_idx = entity_to_idx
        self.entities = entities

        self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        self.model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

    def predict(self, query_phrase, top_k=QUERY_TOP_K):
        query_emb = self.encode_phrase(query_phrase)

        results = self.index.query(
            vector=query_emb.tolist(),
            top_k=top_k, 
            includes_values=INCLUDE_PINCONE_METADATA)
        similar_entities = [
            {
                'entity': match['id'],
                'score': match['score'],
                'metadata': match.get('metadata', {})
            }
            for match in results['matches']
        ]
        return similar_entities

    def compute_similarity(self, phrase, entity):
        inputs = self.tokenizer([phrase, entity], return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)  # Mean pooling to get sentence embeddings
        similarity = torch.nn.functional.cosine_similarity(
            embeddings[0], embeddings[1], dim=0
        ).item()
        return similarity

    def encode_phrase(self, phrase):
        """
        Not an exact science here but it's a good starting point. 
        take the phrase and find the similarity between it and each entity in the graph
        and then take a weighted average of the entity embeddings
        """
        entity_similarities = []
        for entity in self.entities:
            similarity = self.compute_similarity(phrase, entity)
            entity_similarities.append(similarity)
        # Normalize similarities to get weights
        weights = torch.tensor(entity_similarities)
        weights = torch.softmax(weights, dim=0)
        # Compute the weighted sum of entity embeddings
        embedding_dim = self.entity_embeddings.shape[1]
        weighted_sum = torch.zeros(embedding_dim)
        for i, entity in enumerate(self.entities):
            entity_idx = self.entity_to_idx[entity]
            entity_emb = torch.tensor(self.entity_embeddings[entity_idx])
            weighted_sum += weights[i] * entity_emb
        return weighted_sum 

def get_query_model(model_name, saved_graph_info_dir):
    assert model_name in analysis_constants.QUERY_MODELS, f"Model name {model_name} not supported"

    if model_name == analysis_constants.PINECONE:
        return PineconeQueryModel(saved_graph_info_dir) 
    else:
        raise ValueError(f"Model name {model_name} not supported")