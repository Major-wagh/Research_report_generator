import os
from llama_cloud.types import CloudDocumentCreate
from pydantic import BaseModel, Field
from typing import List
from llama_cloud.client import LlamaCloud
from llama_index.core.prompts import PromptTemplate
from llama_index.core.async_utils import run_jobs



class Metadata(BaseModel):
    """Output containing the authors' names, companies, and general AI tags."""
    author_names: List[str] = Field(..., description="List of author names of the paper. Empty list if unavailable.")
    author_companies: List[str] = Field(..., description="List of author companies. Empty list if unavailable.")
    ai_tags: List[str] = Field(..., description="List of general AI tags. Empty list if unavailable.")


def create_llamacloud_pipeline(pipeline_name, embedding_config, transform_config, data_sink_id=None):
    """Function to create a pipeline in LlamaCloud."""
    client = LlamaCloud(token=os.getenv("LLAMA_CLOUD_API_KEY"))

    pipeline = {
        'name': pipeline_name,
        'transform_config': transform_config,
        'embedding_config': embedding_config,
        'data_sink_id': data_sink_id
    }

    pipeline = client.pipelines.upsert_pipeline(request=pipeline)
    return client, pipeline


async def get_papers_metadata(text, llm):
    """Function to get metadata from the given paper."""
    prompt_template = PromptTemplate("""Generate author names, companies, and top 3 AI tags for the research paper.

    Research Paper:

    {text}""")

    metadata = await llm.astructured_predict(
        Metadata,
        prompt_template,
        text=text,
    )
    return metadata


async def get_document_upload(document, llm):
    """Prepare a document for upload with metadata extraction."""
    text_for_metadata_extraction = document[0].text + document[1].text + document[2].text
    full_text = "\n\n".join([doc.text for doc in document])
    metadata = await get_papers_metadata(text_for_metadata_extraction, llm)

    return CloudDocumentCreate(
        text=full_text,
        metadata={
            'author_names': metadata.author_names,
            'author_companies': metadata.author_companies,
            'ai_tags': metadata.ai_tags
        }
    )


async def upload_documents(client, documents, pipeline_id, llm):
    """Function to upload documents to the cloud."""
    extract_jobs = [get_document_upload(document, llm) for document in documents]
    document_upload_objs = await run_jobs(extract_jobs, workers=4)
    client.pipelines.create_batch_pipeline_documents(pipeline_id, request=document_upload_objs)
