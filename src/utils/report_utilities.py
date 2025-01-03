import re

def extract_title(outline):
    """Function to extract the title from the first line of the outline."""
    first_line = outline.strip().split('\n')[0]
    return first_line.strip('# ').strip()

def generate_query_with_llm(llm, title, section, subsection):
    """Function to generate a query for a report using LLM."""
    prompt = (
        f"Generate a research query for a report on {title}. "
        f"The query should be for the subsection '{subsection}' under the main section '{section}'. "
        "The query should guide the research to gather relevant information for this part of the report. "
        "The query should be clear, short, and concise."
    )
    response = llm.complete(prompt)
    return str(response).strip()

def classify_query(llm, query):
    """Function to classify the query as either 'LLM' or 'INDEX' based on the query content."""
    prompt = f"""Classify the following query as either "LLM" if it can be answered directly by a large language model with general knowledge, or "INDEX" if it likely requires querying an external index or database for specific or up-to-date information.

    Query: "{query}"

    Consider the following:
    1. If the query asks for general knowledge, concepts, or explanations, classify as "LLM".
    2. If the query asks for specific facts, recent events, or detailed information that might not be in the LLM's training data, classify as "INDEX".
    3. If unsure, err on the side of "INDEX".

    Classification:"""
    classification = str(llm.complete(prompt)).strip().upper()

    if classification not in ["LLM", "INDEX"]:
        classification = "INDEX"  # Default to INDEX if the response is unclear

    return classification

def parse_outline_and_generate_queries(llm, outline):
    """Function to parse the outline and generate queries for each section and subsection."""
    lines = outline.strip().split('\n')
    title = extract_title(outline)
    current_section = ""
    queries = {}

    for line in lines[1:]:  # Skip the title line
        if line.startswith('## '):
            current_section = line.strip('# ').strip()
            queries[current_section] = {}
        elif re.match(r'^\d+\.\d+\.', line):
            subsection = line.strip()
            query = generate_query_with_llm(llm, title, current_section, subsection)
            classification = classify_query(llm, query)
            queries[current_section][subsection] = {"query": query, "classification": classification}

    # Handle sections without subsections
    for section in queries:
        if not queries[section]:
            query = generate_query_with_llm(llm, title, section, "General overview")
            queries[section]["General"] = {"query": query, "classification": "LLM"}

    return queries
