# Knowledge graph analysis

This directory includes code to analyze knowledge graphs and specifically contains the following capabilites. 
- Embed a knowledge graph and produce entity embeddings. 
- Visualize the embeddings 
- Predict new triples in the knowledge graph 
- Query the knowledge graph using the result embeddings. 


**NB** Currently there is a constant, TEST_MODE=True in analysis_constants.py. This sets the two 
directories DATA_BASE_PATH (where we store analysis results) KNOWLEDGE_BASE_PATH where the json knowledge graphs live. Test mode will set
these two to be test_data/ and test_data/ within test_dir/ but you can set these to be other directories and set TEST_MODE = False if you do 

## Embedding knowledge graphs. embed_graphs.py 

command python embed_graph.py [-f FILE | -d DIRECTORY] [-o OUTPUT] [-m MODEL]
-f, --file
Path to a json file 

-d, --directory
Path to a folder containing multiple json files. Default ouput file will be KNOWLEDGE_BASE_PATH/

-o, --output (optional)
Path to save the embedded graphs. Default: DATA_BASE_PATH/embeddings 

-m, --model (optional)
Embedding model to use. Default: TransR. Choices: TransR.

These embeds our knowledge graph and clusters the entities based on proximity in the embedding space

The ouput for each graph is a directory i.e embedding_directory that contains several files 
    - entity embeddings     
    - relation embeddings 
    - graph_data 
        - entitities  (set of entities) 
        - relation  (set of edge types)
        - entity to index map 
        - relation to index map 
        - cluster assignments 
        
we save these together since they are useful for downstream analysis tasks. 

Running "python embed_graphs.py" will embed graphs in test_kgs i.e social.json and politics.json

## Visualizing the embeddings. visualize_embeddings.py

python visualize_embeddings.py [-d DIRECTORY] [-o OUTPUT]

The  directory here is a directory is a directory to embeddings_directories for potentially multiple 
graphs. The script will visualize all of them. 
This defaults to DATA_BASE_PATH/embeddings

The ouput directory defaults to DATA_BASE_PATH/plots

Running "python visualize_embeddings.py" will plot embeddings for social.json and politics.json 


## Predicting new triples. 

python graph_imputation.py [-d DIRECTORY] [-o OUTPUT]

The  directory here is a directory is a directory to embeddings_directories for potentially multiple 
graphs. The script will predict new triples for all of them. 
This defaults to DATA_BASE_PATH/embeddings

The ouput directory defaults to DATA_BASE_PATH/imputation
Running "python visualize_embeddings.py" will predict new triples for for social.json and politics.json. The number of new triples is based on the constant K_NEW_TRIPLES. 

Running "python graph_imputation.py" will predict K_NEW_TRIPLES for both social.json and politics.json 

The output for each graph here is a json file of the format 

"{
    "original_triples": [...]
    "predicted_triples": [...]
}"

## Querying the KG 

python query_kg.py -f /abs/path/to/file_with_query.txt [-d DIRECTORY] [-o OUTPUT]

The  directory here is a directory is a directory to embeddings_directories for potentially multiple 
graphs. The script will predict new triples for all of them. 
This defaults to DATA_BASE_PATH/embeddings

The ouput directory defaults to DATA_BASE_PATH/imputation
Running "python visualize_embeddings.py" will predict new triples for for social.json and politics.json. For each query, we get K_TOP_RESPONSE_ENTITIES, which is the topK responses to the query. 

The input file should be a txt file where each line is a phrase. the ouput for each knowledge graph is a json file 

















