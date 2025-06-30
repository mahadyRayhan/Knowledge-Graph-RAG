# graph_setup.py

import os
import re
import sys
import io
import pandas as pd
from dotenv import load_dotenv
from langchain_community.graphs import Neo4jGraph
from langchain.chains import GraphCypherQAChain
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

print("--- Initializing Graph and Chain Setup ---")

# --- 1. Load Environment Variables ---
load_dotenv()
gemini_api = os.getenv("GOOGLE_API_KEY")
neo4j_url = os.getenv("NEO4J_CONNECTION_URL")
neo4j_user = os.getenv("NEO4J_USER")
neo4j_password = os.getenv("NEO4J_PASSWORD")
neo4j_db = os.getenv("NEO4J_SCHOLAR")

# --- 2. Data Loading and Preprocessing Function ---
def load_and_preprocess_data():
    print("Loading and preprocessing data...")
    scholar_data = pd.read_csv("data/scholer_recommendation.csv")
    scholar_data = scholar_data.drop(columns=["Abstract", "Keywords"], axis=1)
    scholar_data = pd.concat([scholar_data.head(80), scholar_data.tail(20)], ignore_index=True)
    scholar_data.rename(columns={'Fields of Study': 'Discipline', 'Category': 'Topic'}, inplace=True)

    scholar_data['Authors_list'] = scholar_data['Authors'].str.split(',')
    scholar_data = scholar_data.explode('Authors_list').reset_index(drop=True)
    scholar_data["Authors"] = scholar_data["Authors_list"]
    scholar_data.drop(["Authors_list"], axis=1, inplace=True)

    scholar_data['Discipline_list'] = scholar_data['Discipline'].str.split(',')
    scholar_data = scholar_data.explode('Discipline_list').reset_index(drop=True)
    scholar_data["Discipline"] = scholar_data["Discipline_list"]
    scholar_data.drop(["Discipline_list"], axis=1, inplace=True)

    scholar_data.drop_duplicates(inplace=True)
    scholar_data.dropna(inplace=True)
    scholar_data.reset_index(drop=True, inplace=True)
    
    scholar_data.rename(columns={'Title': 'Paper Title', 'Authors': 'Author', 'Year': 'Year Published'}, inplace=True)
    scholar_data = scholar_data.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    
    unique_authors = scholar_data['Author'].unique()
    author_to_id = {author: f"{idx:09d}" for idx, author in enumerate(sorted(unique_authors))}
    scholar_data['pi_id'] = scholar_data['Author'].map(author_to_id)
    
    print("Data preprocessing complete.")
    return scholar_data

# --- 3. Neo4j Graph Population Logic ---
def populate_graph(graph, data):
    print("Populating Neo4j graph...")
    # Clear existing graph
    graph.query("MATCH (n) DETACH DELETE n")

    for index, row in data.iterrows():
        # Using a single, more efficient query with MERGE for all nodes and relationships
        query = """
        MERGE (p:Paper {title: $paper_title})
        ON CREATE SET p.year = $year_published, p.citations = $citations, p.topic = $topic
        MERGE (a:Author {name: $author})
        MERGE (d:Discipline {name: $discipline})
        MERGE (v:Venue {name: $venue})
        MERGE (a)-[:AUTHORED]->(p)
        MERGE (p)-[:BELONGS_TO]->(d)
        MERGE (p)-[:PUBLISHED_IN]->(v)
        """
        parameters = {
            'paper_title': row['Paper Title'],
            'year_published': row['Year Published'],
            'citations': row['Citations'],
            'topic': row['Topic'],
            'author': row['Author'],
            'discipline': row['Discipline'],
            'venue': row['Venue']
        }
        graph.query(query, parameters)
    print("Graph population complete.")

def check_and_populate_graph(graph):
    # Check if the graph has any data
    result = graph.query("MATCH (n) RETURN count(n) as count")
    if result[0]['count'] == 0:
        print("Graph is empty. Proceeding with data loading and population.")
        scholar_data = load_and_preprocess_data()
        populate_graph(graph, scholar_data)
    else:
        print(f"Graph already contains {result[0]['count']} nodes. Skipping population.")

# --- 4. LangChain Setup ---
# Initialize Neo4j connection
graph = Neo4jGraph(neo4j_url, neo4j_user, neo4j_password, database=neo4j_db)

# Check graph state and populate if necessary
check_and_populate_graph(graph)

# Prompt Examples and Templates
example = [
    {
        "question": "List all papers authored by 'Han Xiao'.",
        "query": "MATCH (a:Author {name: 'Han Xiao'})-[:AUTHORED]->(p:Paper) RETURN p.title AS PapersAuthoredByHanXiao",
    },
    {
        "question": "Which papers belong to the 'Computer Science' discipline?",
        "query": "MATCH (p:Paper)-[:BELONGS_TO]->(d:Discipline {name: 'Computer Science'}) RETURN p.title AS PapersInComputerScience Limit 5"
    },
    {
        "question": "What are the papers published in 'Nature' in the year 2018?",
        "query": "MATCH (p:Paper)-[:PUBLISHED_IN]->(v:Venue {name: 'Nature'}) WHERE p.year = 2018 RETURN p.title AS PapersPublishedInNature2018"
    },
    {
        "question": "How many papers did 'Jianmin Chen' author?",
        'query': "MATCH (a:Author {name: 'Jianmin Chen'})-[:AUTHORED]->(p:Paper) RETURN COUNT(p) AS NumberOfPapersAuthoredByJianminChen"
    },
    {
        "question": "List all authors who have published papers in the topic 'Machine Learning'.",
        "query": "MATCH (a:Author)-[:AUTHORED]->(p:Paper {topic: 'Machine Learning'}) RETURN DISTINCT a.name AS AuthorsInMachineLearning Limit 5"
    },
    {
        'question': "What are the most cited papers in 'Mathematics'?",
        'query': "MATCH (p:Paper)-[:BELONGS_TO]->(d:Discipline {name: 'Mathematics'}) RETURN p.title AS PapersInComputerScience Limit 5"
    },
    {
        'question': "What are the most cited papers in 'Materials Science' discipline?",
        'query': "MATCH (p:Paper)-[:BELONGS_TO]->(d:Discipline {name: 'Materials Science'}) RETURN p.title AS Paper, p.citations AS Citations ORDER BY Citations DESC LIMIT 5"
    },
    {
        'question': "Which venues have published papers in the 'Network Science' topic?",
        'query': "MATCH (p:Paper {topic: 'Network Science'})-[:PUBLISHED_IN]->(v:Venue) RETURN DISTINCT v.name AS VenuesForNetworkScience LIMIT 5"
    },
    {
        'question': "I am 'Han Xiao' conducts research in 'Computer Science' and 'Machine Learning'. Which professors should he collaborate with?",
        'query': "MATCH (a:Author {name: 'Han Xiao'})-[:AUTHORED]->(p:Paper)-[:BELONGS_TO]->(d:Discipline) WHERE d.name = 'Computer Science' OR p.topic = 'Machine Learning' WITH DISTINCT d AS Discipline, p.topic AS Topic MATCH (other:Author)-[:AUTHORED]->(:Paper)-[:BELONGS_TO]->(d) WHERE other.name <> 'Han Xiao' RETURN DISTINCT other.name AS PotentialCollaborators LIMIT 5"
    },
    {
        'question': "I am 'Han Xiao'. Which researchers I collaborated with before?",
        'query': "MATCH (a1:Author {name: 'Han Xiao'})-[:AUTHORED]->(p:Paper)<-[:AUTHORED]-(a2:Author) WHERE a1 <> a2 RETURN DISTINCT a2.name"
    },
    {
        'question': "I am 'Han Xiao'. Which new researchers should I collaborate with for future work?",
        'query': "MATCH (a1:Author {name: 'Han Xiao'})-[:AUTHORED]->(p:Paper)-[:BELONGS_TO]->(d:Discipline)<-[:BELONGS_TO]-(p2:Paper)<-[:AUTHORED]-(a2:Author) WHERE a1 <> a2 RETURN a2.name, COUNT(p2) AS collaborations ORDER BY collaborations DESC"
    },
    {
        'question': "I am 'Kashif Rasul'. I have some workes in 'Mathematics' and want to expand my research in this field. Which researchers should I collaborate with based on papers related to 'Mathematics'?",
        'query': "MATCH (a:Author {name: 'Kashif Rasul'})-[:AUTHORED]->(p:Paper)-[:BELONGS_TO]->(d:Discipline) WHERE d.name = 'Mathematics' WITH DISTINCT d AS Discipline MATCH (other:Author)-[:AUTHORED]->(:Paper)-[:BELONGS_TO]->(d) WHERE other.name <> 'Kashif Rasul' RETURN DISTINCT other.name AS PotentialCollaborators"
    }
]


cypher_generation_prompt = PromptTemplate(
    template="""Based on the schema, write a Cypher query to answer the question.
    Schema: {schema}
    Example questions and queries: {example}
    **Important**:
    - Always filter by specific properties in the question when provided.
    - Provide meaningful aliases in the RETURN statement.
    Question: {question}
    Query:""",
    input_variables=["schema", "question", "example"],
)

qa_prompt = PromptTemplate(
    template="""Based on the Cypher query results, answer the question.
    Question: {question}
    Results: {context}
    Give a clear, direct, and human-friendly answer using the data from the results. 
    If it's a list, combine all items and summarize.
    Answer:""",
    input_variables=["question", "context"],
)

# Initialize LLM and Chain
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=gemini_api, temperature=0)
chain = GraphCypherQAChain.from_llm(
    llm=llm,
    graph=graph,
    verbose=True,
    cypher_generation_prompt=cypher_generation_prompt,
    qa_prompt=qa_prompt,
    allow_dangerous_requests=True,
)

# --- 5. Main Query Function for the App ---
def clean_ansi(text):
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', text).strip()

def query_graph(question: str) -> dict:
    """
    Takes a user question, queries the graph, and returns structured results.
    This function will be imported and used by the Flask app.
    """
    try:
        output_buffer = io.StringIO()
        original_stdout = sys.stdout
        sys.stdout = output_buffer

        # Run the chain. Pass examples directly if the chain supports it or format into prompt.
        result = chain.invoke({"query": question, "example": example})

        sys.stdout = original_stdout
        captured_output = output_buffer.getvalue()
        
        # Extract details from verbose output
        cypher_query = "Could not extract Cypher query."
        full_context = "Could not extract context."
        
        if 'Generated Cypher:' in captured_output:
            cypher_part = captured_output.split('Generated Cypher:')[1]
            cypher_query = clean_ansi(cypher_part.split('\n\n')[0])
        
        if 'Full Context:' in captured_output:
            context_part = captured_output.split('Full Context:')[1]
            full_context = clean_ansi(context_part.split('\n\n\x1b[0m>')[0])

        return {
            'answer': result.get('result', 'No answer found.'),
            'generated_query': cypher_query,
            'full_context': full_context,
            'error': None
        }
    except Exception as e:
        print(f"Error during graph query: {str(e)}")
        return {
            'answer': "Sorry, I encountered an error while processing your request.",
            'generated_query': None,
            'full_context': None,
            'error': str(e)
        }

print("--- Setup Complete. Ready for Queries. ---")