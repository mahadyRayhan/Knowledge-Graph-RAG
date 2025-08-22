# populate_knowledge_graph.py

import os
import json
from dotenv import load_dotenv
from langchain_neo4j import Neo4jGraph
import re
from collections import defaultdict

# --- Configuration ---
load_dotenv('../.env')
NEO4J_URL = os.getenv("NEO4J_CONNECTION_URL")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DB = os.getenv("NEO4J_SCHOLAR")

S2_DATA_DIR = "s2_author_data"

# --- KEYWORD IMPROVEMENT ---
# A curated list of high-value academic and technical keywords.
# This list is the key to creating a clean, useful graph.
# We will only create keyword nodes for words that are on this list.
DOMAIN_KEYWORDS = {
    # Computer Science & AI
    'llm', 'machine learning', 'data science', 'deep learning', 'neural network',
    'computer vision', 'nlp', 'natural language processing', 'robotics', 'algorithm',
    'cybersecurity', 'blockchain', 'software engineering', 'database', 'cloud computing',
    # Physical Sciences & Engineering
    'quantum computing', 'semiconductor', 'nanotechnology', 'material science',
    'chemical engineering', 'petrochemical', 'thermal', 'thermodynamics', 'fluid dynamics',
    'photonics', 'optics', 'acoustics', 'mechanical engineering', 'electrical engineering',
    # Life Sciences & Medicine
    'bioinformatics', 'genomics', 'proteomics', 'drug discovery', 'neuroscience',
    'immunology', 'microbiology', 'molecular biology', 'biochemistry', 'genetics',
    'pharmacology', 'virology', 'oncology', 'cardiology',
    # Environmental & Earth Sciences
    'climate change', 'sustainability', 'geology', 'hydrology', 'oceanography',
    'ecology', 'sediment', 'erosion', 'bedrock', 'hillslope', 'meteorology',
    # Social Sciences & Other
    'economics', 'psychology', 'sociology', 'linguistics', 'statistics', 'mathematics'
}

def extract_domain_keywords(text: str) -> set:
    """
    Extracts only the high-value, pre-defined domain keywords from a block of text.
    """
    if not isinstance(text, str): return set()
    # Use word boundaries (\b) to match whole words only
    found_keywords = {keyword for keyword in DOMAIN_KEYWORDS if re.search(r'\b' + re.escape(keyword) + r'\b', text, re.IGNORECASE)}
    return found_keywords

def run_graph_ingestion():
    """
    Reads all fetched S2 JSON data and populates the Neo4j Knowledge Graph.
    """
    print("--- Starting Knowledge Graph Population Process ---")
    try:
        graph_db = Neo4jGraph(url=NEO4J_URL, username=NEO4J_USER, password=NEO4J_PASSWORD, database=NEO4J_DB)
        print("Successfully connected to Neo4j.")
    except Exception as e:
        print(f"Error connecting to Neo4j: {e}")
        return
    
    # Optional: Clean the database before a full re-ingestion
    print("Clearing existing database...")
    graph_db.query("MATCH (n) DETACH DELETE n")
    
    print("Creating database constraints for performance...")
    graph_db.query("CREATE CONSTRAINT researcher_s2_id IF NOT EXISTS FOR (r:Researcher) REQUIRE r.s2_author_id IS UNIQUE")
    graph_db.query("CREATE CONSTRAINT paper_s2_id IF NOT EXISTS FOR (p:Paper) REQUIRE p.s2_paper_id IS UNIQUE")
    graph_db.query("CREATE CONSTRAINT affiliation_name IF NOT EXISTS FOR (a:Affiliation) REQUIRE a.name IS UNIQUE")
    graph_db.query("CREATE CONSTRAINT keyword_name IF NOT EXISTS FOR (k:Keyword) REQUIRE k.name IS UNIQUE")

    json_files = [f for f in os.listdir(S2_DATA_DIR) if f.endswith('.json')]
    total_files = len(json_files)
    print(f"Found {total_files} researcher data files to ingest.")

    for i, filename in enumerate(json_files):
        filepath = os.path.join(S2_DATA_DIR, filename)
        with open(filepath, 'r') as f:
            data = json.load(f)

        print(f"\n[{i+1}/{total_files}] Ingesting data for: {data.get('name')}")

        s2_author_id = data.get('s2_author_id') or data.get('authorId')
        if not s2_author_id:
            print(f"  -> SKIPPING: Could not find author ID in {filename}")
            continue

        # Create/update the Researcher node
        graph_db.query("""
            MERGE (r:Researcher {s2_author_id: $s2_id})
            ON CREATE SET r.name = $name, r.nsf_pi_id = $nsf_id
            SET r.hIndex = $h_index, r.citationCount = $citation_count, r.name = $name
        """, {
            "s2_id": s2_author_id, "name": data.get('name'), "nsf_id": data.get('nsf_pi_id'),
            "h_index": data.get('hIndex'), "citation_count": data.get('citationCount')
        })

        # --- THE FIX FOR AFFILIATIONS ---
        # Conditionally create the Affiliation and link it, now correctly separated.
        if data.get('affiliations') and data['affiliations']:
            primary_affiliation = data['affiliations'][0]
            graph_db.query("""
                MATCH (r:Researcher {s2_author_id: $s2_id})
                MERGE (a:Affiliation {name: $affiliation_name})
                MERGE (r)-[:AFFILIATED_WITH]->(a)
            """, {
                "s2_id": s2_author_id,
                "affiliation_name": primary_affiliation
            })

        for paper in data.get('papers', []):
            s2_paper_id = paper.get('s2_paper_id') or paper.get('paperId')
            if not s2_paper_id: continue
            
            graph_db.query("""
                MATCH (author:Researcher {s2_author_id: $s2_author_id})
                MERGE (p:Paper {s2_paper_id: $s2_paper_id})
                ON CREATE SET p.title = $title, p.year = $year, p.venue = $venue, p.citationCount = $citations
                MERGE (author)-[:AUTHORED]->(p)
            """, {
                "s2_paper_id": s2_paper_id, "title": paper.get('title'),
                "year": paper.get('year'), "venue": paper.get('venue'),
                "citations": paper.get('citationCount'), "s2_author_id": s2_author_id
            })
            
            # Use the new, smarter keyword extractor
            keywords = extract_domain_keywords(paper.get('abstract'))
            if keywords:
                print(f"  -> Found relevant keywords: {keywords}")
                for keyword in keywords:
                    graph_db.query("""
                        MATCH (p:Paper {s2_paper_id: $s2_paper_id})
                        MERGE (k:Keyword {name: $keyword})
                        MERGE (p)-[:HAS_KEYWORD]->(k)
                    """, {"s2_paper_id": s2_paper_id, "keyword": keyword})

        for co_author in data.get('co_authors', []):
             co_author_id = co_author.get('s2_author_id') or co_author.get('authorId')
             if not co_author_id: continue
             graph_db.query("""
                MATCH (main_author:Researcher {s2_author_id: $main_author_id})
                MERGE (co_author:Researcher {s2_author_id: $co_author_id})
                ON CREATE SET co_author.name = $co_author_name
                CALL apoc.merge.relationship(main_author, 'COLLABORATED_WITH', {}, {}, co_author) YIELD rel
                RETURN rel
            """, {
                "main_author_id": s2_author_id,
                "co_author_id": co_author_id,
                "co_author_name": co_author.get('name')
            })

    print("\n--- Knowledge Graph Population Complete ---")

if __name__ == "__main__":
    run_graph_ingestion()