import os
from dotenv import load_dotenv
from llama_index.indices.managed.llama_cloud import LlamaCloudIndex

def setup_query_engine(llm):
    # Load environment variables
    load_dotenv()

    # Ensure the API key is loaded
    llama_api_key = os.getenv('LLAMA_CLOUD_API_KEY')
    if not llama_api_key:
        raise ValueError("LLAMA_CLOUD_API_KEY is missing. Check your .env file.")

    # Connect to the existing index
    print("Connecting to LlamaCloud index...")
    index = LlamaCloudIndex(
        name="report_generation",
        project_name="Default",
        api_key=llama_api_key
    )
    print("Index connected successfully.")

    # Build the QueryEngine
    print("Setting up QueryEngine...")
    query_engine = index.as_query_engine(
        llm=llm,
        dense_similarity_top_k=10,
        sparse_similarity_top_k=10,
        alpha=0.5,
        enable_reranking=True,
        rerank_top_n=5,
        retrieval_mode="chunks"
    )
    print("QueryEngine setup complete.")
    
    return query_engine
