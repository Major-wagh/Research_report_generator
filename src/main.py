import os
import nest_asyncio
import asyncio # Required for asynchronous execution
import argparse  
from dotenv import load_dotenv
from pathlib import Path
from utils.query_engine_setup import setup_query_engine
from utils.list_pdf_files import list_pdf_files
from utils.document_parse import DocumentParser
from utils.download_papers import ArxivClient
from utils.utils import create_llamacloud_pipeline, upload_documents
from llama_index.llms.groq import Groq
from utils.ReportGenerationAgent import ReportGenerationAgent
from llm.llm_client import LLMClient


async def main():
    # Load environment variables from .env file
    load_dotenv()

    # Get API keys from environment variables
    llama_api_key = os.getenv('LLAMA_CLOUD_API_KEY')
    groq_api_key = os.getenv('GROQ_API_KEY')
    cohere_api_key = os.getenv('COHERE_API_KEY')

    if not llama_api_key or not groq_api_key or not cohere_api_key:
        raise ValueError("API keys are missing. Check your .env file.")

    print("API keys loaded successfully.")

    # Initialize GroqClient
    llm=LLMClient()
    llm=llm.llm

    # Directory for downloaded PDFs
    download_directory = './data/raw'
    os.makedirs(download_directory, exist_ok=True)

    # Step 1: Download papers
    arxiv_client = ArxivClient(max_results_per_topic=max_results_per_topic, download_dir=download_directory)
    arxiv_client.download_papers(topics)

    # Step 2: List PDFs
    pdf_files = list_pdf_files(download_directory)
    print("Downloaded PDFs:", pdf_files)

    if not pdf_files:
        print("No PDF files found in the directory.")
        return

    # Step 3: Parse PDFs
    document_parser = DocumentParser(result_type="markdown", num_workers=4, api_key=llama_api_key)
    documents = document_parser.parse_files(pdf_files, download_directory)

    # Step 4: Define embedding and transform configurations
    embedding_config = {
        'type': 'COHERE_EMBEDDING',
        'component': {
            'api_key': cohere_api_key,  # Editable
            'model_name': 'embed-english-v3.0'  # Editable
        }
    }

    transform_config = {
        'mode': 'auto',
        'config': {
            'chunk_size': 1024,
            'chunk_overlap': 20
        }
    }

    # Step 5: Create LlamaCloud pipeline
    print("Creating LlamaCloud pipeline...")
    pipeline_name = "report_generation"
    client, pipeline = create_llamacloud_pipeline(pipeline_name, embedding_config, transform_config)
    print(f"Pipeline '{pipeline_name}' created successfully.")

    # Step 6: Asynchronously upload documents to the created pipeline
    print("Uploading documents...")
    await upload_documents(client, documents, pipeline.id, llm)
    print("All documents uploaded successfully.")

    # Step 7: Setup QueryEngine
    print("Initializing QueryEngine...")
    query_engine = setup_query_engine(llm=llm)  # Assuming setup_query_engine returns a working query engine

    outline = """
    # Research Paper Report on RAG - Retrieval Augmented Generation and Agentic World.

    ## 1. Introduction

    ## 2. Retrieval Augmented Generation (RAG) and Agents
    2.1. Fundamentals of RAG and Agents.
    2.2. Current State and Applications

    ## 3. Latest Papers:
    3.1. HEALTH-PARIKSHA: Assessing RAG Models for Health Chatbots in Real-World Multilingual Settings
    3.2. MIRAGE-Bench: Automatic Multilingual Benchmark Arena for Retrieval-Augmented Generation Systems
    3.3. VLM-Grounder: A VLM Agent for Zero-Shot 3D Visual Grounding

    ## 4. Conclusion:
    """

    # Step 8: Instantiate and run the ReportGenerationAgent
    print("Starting report generation using the ReportGenerationAgent...")

    agent = ReportGenerationAgent(query_engine=query_engine, llm=llm, verbose=True, timeout=1200.0)

    # Call the agent to generate the report based on the outline
    report = await agent.run(outline=outline)

    # Output the final report
    print("Generated Report:\n", report['response'])

if __name__ == "__main__":
    nest_asyncio.apply()
    parser = argparse.ArgumentParser(description="Run the paper download and report generation process.")
    parser.add_argument('topics', type=str, help="Comma-separated list of topics (e.g., Rag,agents)")
    parser.add_argument('max_results_per_topic', type=int, help="Maximum number of results per topic (e.g., 3)")
    
    args = parser.parse_args()

    topics = args.topics.split(',')  # Convert comma-separated string to a list
    max_results_per_topic = args.max_results_per_topic
    asyncio.run(main())
