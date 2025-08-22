# graph_setup.py

# This print statement is a new diagnostic tool. If you don't see this line,
# you are still running an old file.
print("--- EXECUTING LATEST graph_setup.py (VERSION WITH ROUTER) ---")

import os
import re
import sys
import io
import pandas as pd
from dotenv import load_dotenv
from typing import List, Dict, Tuple, Optional, Any

# New imports for the Router and updated LangChain package
import google.generativeai as genai
# Updated import to fix the deprecation warning
from langchain_neo4j import Neo4jGraph
from langchain.chains import GraphCypherQAChain
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

# --- 1. Load Environment Variables and Configure APIs ---
load_dotenv()
gemini_api = os.getenv("GOOGLE_API_KEY")
neo4j_url = os.getenv("NEO4J_CONNECTION_URL")
neo4j_user = os.getenv("NEO4J_USER")
neo4j_password = os.getenv("NEO4J_PASSWORD")
neo4j_db = os.getenv("NEO4J_SCHOLAR")


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
genai.configure(api_key=gemini_api)

# --- 2. Data Processing and Graph Population ---
def load_and_preprocess_data():
    """ This function contains the detailed data processing pipeline. """
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
    scholar_data = scholar_data.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    scholar_data.drop_duplicates(inplace=True)
    scholar_data.dropna(inplace=True)
    scholar_data.reset_index(drop=True, inplace=True)
    scholar_data.rename(columns={'Title': 'Paper Title', 'Authors': 'Author', 'Year': 'Year Published'}, inplace=True)
    print(f"Data preprocessing complete. Shape: {scholar_data.shape}")
    return scholar_data

def populate_graph(graph, data):
    print("Populating Neo4j graph with optimized query...")
    graph.query("MATCH (n) DETACH DELETE n")
    query = """
    UNWIND $rows AS row
    MERGE (p:Paper {title: row.`Paper Title`})
    ON CREATE SET p.year = row.`Year Published`, p.citations = row.Citations, p.topic = row.Topic
    MERGE (a:Author {name: row.Author})
    MERGE (d:Discipline {name: row.Discipline})
    MERGE (v:Venue {name: row.Venue})
    MERGE (a)-[:AUTHORED]->(p)
    MERGE (p)-[:BELONGS_TO]->(d)
    MERGE (p)-[:PUBLISHED_IN]->(v)
    """
    rows = data.to_dict('records')
    graph.query(query, {'rows': rows})
    graph.refresh_schema()
    print("Graph population and schema refresh complete.")

def check_and_populate_graph(graph):
    """Checks if the graph is empty and populates it if needed."""
    result = graph.query("MATCH (n) RETURN count(n) as count")
    if result[0]['count'] == 0:
        print("Graph is empty. Proceeding with data loading and population.")
        scholar_data = load_and_preprocess_data()
        populate_graph(graph, scholar_data)
    else:
        print(f"Graph already contains {result[0]['count']} nodes. Skipping population.")
        graph.refresh_schema()


# --- 3. Define the System Component Classes ---
class ScholarKGQA:
    def __init__(
        self,
        google_api_key: Optional[str] = None,
        neo4j_url: Optional[str] = None,
        neo4j_user: Optional[str] = None,
        neo4j_password: Optional[str] = None,
        neo4j_db: Optional[str] = None,
        llm_model_tag: str = "gemini-2.5-pro-preview-03-25",
        llm_temperature: float = 0.0,
        example: List[Dict[str, str]] = example,
        verbose: bool = True,
        allow_dangerous_requests: bool = True
    ):
        # This internal logic correctly uses os.getenv, fitting with the external loading pattern
        self.google_api_key = google_api_key or os.getenv("GOOGLE_API_KEY")
        self.neo4j_url = neo4j_url or os.getenv("NEO4J_CONNECTION_URL")
        self.neo4j_user = neo4j_user or os.getenv("NEO4J_USER")
        self.neo4j_password = neo4j_password or os.getenv("NEO4J_PASSWORD")
        self.neo4j_db = neo4j_db or os.getenv("NEO4J_SCHOLAR") # Ensure env var name matches
        self.llm_model_tag = llm_model_tag
        self.verbose = verbose
        self.example = example
        self.ALLOW_DANGEROUS_REQUEST = allow_dangerous_requests
        self.llm_temperature = llm_temperature
        self.graph = Neo4jGraph(self.neo4j_url,self. neo4j_user, self.neo4j_password, database=self.neo4j_db)
        
        if not all([self.google_api_key, self.neo4j_url, self.neo4j_user, self.neo4j_password, self.neo4j_db]):
            raise ValueError("Missing required configuration (API Key or Neo4j credentials/URL/DB). "
                             "Ensure environment variables are set or pass arguments directly.")
            
        self.llm = ChatGoogleGenerativeAI(model=llm_model_tag, google_api_key=self.google_api_key, temperature=self.llm_temperature)
        
        self.cypher_generation_prompt = PromptTemplate(
            template="""Based on the schema, write a Cypher query to answer the question.

            The question may ask about:
            - Authors and their research fields
            - Publication venues and trends
            - Paper citations and collaborations
            - Discipline for authors and papers
            - Recommendations for collaborations or venues

            Schema:
            {schema}

            Example questions and queries:
            {example}

            **Important**:
            - Always filter by specific properties in the question when provided, such as `category` for papers or `name` for authors.
            - Ensure the query aligns precisely with the requested category, author, or venue.
            - When counting or aggregating, provide meaningful aliases like `VenueName`, `PaperCount`, or `AuthorName`.
            - Do not include irrelevant nodes or relationships in the query.

            Question: {question}
            Query:""",
            input_variables=["schema", "question", "example"],
        )
        
        self.qa_prompt = PromptTemplate(
            template="""Based on the Cypher query results, answer the question.
            Question: {question}
            Results: {context}
            Give a clear, direct, and human-friendly answer using the data from the results. 
            If it's a list, combine all items and summarize. For example, for authors or papers, list them in a human-readable format.
            Answer:""",
            input_variables=["question", "context"],
        )
        
        self.chain = GraphCypherQAChain.from_llm(
            llm=self.llm,
            graph=self.graph,  # Your Neo4j graph object
            verbose=True,
            cypher_generation_prompt=self.cypher_generation_prompt,
            qa_prompt=self.qa_prompt,
            allow_dangerous_requests=self.ALLOW_DANGEROUS_REQUEST,
        )
        
    def clean_ansi(self, text):
        # Remove ANSI escape codes
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        return ansi_escape.sub('', text).strip()

    def query(self, question):
        try:
            # Create a string buffer to capture the output
            output_buffer = io.StringIO()
            original_stdout = sys.stdout
            # sys.stdout = output_buffer

            # Explicitly generate the Cypher query first using the prompt
            generated_query = self.cypher_generation_prompt.format(
                schema=self.graph.schema,  # Ensure dynamic schema usage
                question=question,
                example=self.example
            )

            # Run the chain with the generated query
            # result = self.chain.run(query=generated_query, question=question)
            input_data = {
                "query": generated_query,
                "question": question
            }
            result = self.chain.invoke(input_data)
            # result = self.chain.invoke({
            #     "query": question,
            #     "example": self.example_queries # Pass examples here
            #     })
            
            # Restore original stdout and get the captured output
            sys.stdout = original_stdout
            output = output_buffer.getvalue()
            
            # Extract Cypher query and context from the captured output
            cypher_query = None
            full_context = None
            
            if 'Generated Cypher:' in output:
                cypher_query = output.split('Generated Cypher:')[1].split('Full Context:')[0].strip()
                cypher_query = self.clean_ansi(cypher_query)
            
            if 'Full Context:' in output:
                full_context = output.split('Full Context:')[1].split('>')[0].strip()
                full_context = self.clean_ansi(full_context)
            
            # print(f"Q: {question}")
            # print(f"A: {result}\n")
            
            return {
                'result': result,
                'cypher_query': cypher_query,
                'full_context': full_context
            }
        except Exception as e:
            print(f"Error: {str(e)}")
            return {
                'result': None,
                'cypher_query': None,
                'full_context': None,
                'error': str(e)
            }
                   
# --- Class for PI Ranking and Analysis ---
class PIRankingAnalysis:
    def __init__(self, ranking_data: pd.DataFrame = None, grouped_data: pd.DataFrame = None, llm_model_name: str = None):
        print(f"Initializing PIRankingAnalysis with model: {llm_model_name}")

class QueryRouter:
    """Routes questions to the appropriate system (KG or PI Analysis)."""
    def __init__(self, kg_system: ScholarKGQA, pi_system: PIRankingAnalysis, router_llm_model_name: str):
        self.kg_system = kg_system
        self.pi_system = pi_system
        self.router_model = genai.GenerativeModel(router_llm_model_name)
    def _get_intent_from_llm(self, question: str) -> str:
        prompt = f"""Classify the user's question into one of the following categories:
            - kg_query: Asking about papers, authors, venues, citations, collaborations, disciplines.
            - pi_recommendation: Asking to recommend a PI for a topic from a list of researchers.
            - influencer_list: Asking to rank or find the best influencer from a list of researchers.
            - influencer_topic: Asking to find the best influencer for a specific research topic.
            - influencer_dept: Asking to find the best influencer within a specific department.
            - unknown: If the question doesn't fit.
            Question: "{question}"
            Category: """
        try:
            safety_settings = [{"category": c, "threshold": "BLOCK_NONE"} for c in ["HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH", "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"]]
            response = self.router_model.generate_content(prompt, safety_settings=safety_settings)
            intent = response.candidates[0].content.parts[0].text.strip().lower()
            intent = re.sub(r'[^\w_]', '', intent)
            valid_intents = ["kg_query", "pi_recommendation", "influencer_list", "influencer_topic", "influencer_dept", "unknown"]
            if intent not in valid_intents: return "kg_query"
            print(f"LLM classified intent as: {intent}")
            return intent
        except Exception as e:
            print(f"Error during intent classification: {e}")
            return "kg_query"
    def _extract_parameters(self, question: str, intent: str) -> Dict[str, Any]:
        params = {}
        if intent == "pi_recommendation":
            id_match = re.search(r"(\[.*?\])", question, re.IGNORECASE)
            if id_match: params["pi_ids"] = self._extract_pi_ids(id_match.group(1))
            topic_match = re.search(r"\s(?:in|topic)\s+[\'\"]?([^\'\"]+)[\'\"]?", question, re.IGNORECASE)
            if topic_match and not topic_match.group(1).strip().startswith('['): params["topic"] = topic_match.group(1).strip()
        elif intent == "influencer_list":
             match = re.search(r"among\s+(?:them|these)\s*:\s*(.*)", question, re.IGNORECASE)
             if match: params["pi_ids"] = self._extract_pi_ids(match.group(1))
        elif intent == "influencer_topic":
             match = re.search(r"topic\s+[\'\"]?([^\'\"]+)[\'\"]?", question, re.IGNORECASE)
             if match: params["topic"] = match.group(1).strip()
        elif intent == "influencer_dept":
             match = re.search(r"in\s+[\'\"]?([^\'\"]+)[\'\"]?\s+department", question, re.IGNORECASE)
             if match: params["department"] = match.group(1).strip()
        return params
    def _extract_pi_ids(self, text: str) -> List[str]:
         match = re.search(r"\[\s*([\'\"]?\s*\d+\s*[\'\"]?(?:\s*,\s*[\'\"]?\d+\s*[\'\"]?)*)\s*\]", text)
         return [item.strip('\'" ') for item in match.group(1).split(',')] if match else []
    def route(self, question: str):
        intent = self._get_intent_from_llm(question)
        if intent == "kg_query": return self.kg_system.query(question)
        params = self._extract_parameters(question, intent)
        if intent == "pi_recommendation" and "pi_ids" in params and "topic" in params: return self.pi_system.recommend_pi(params["pi_ids"], params["topic"])
        if intent == "influencer_list" and "pi_ids" in params: return self.pi_system.rank_influencers_by_list(params["pi_ids"])
        if intent == "influencer_topic" and "topic" in params: return self.pi_system.find_influencer_by_criterion("topic", params["topic"])
        if intent == "influencer_dept" and "department" in params: return self.pi_system.find_influencer_by_criterion("department", params["department"])
        return self.kg_system.query(question)

# --- 4. Initialize Global Instances ---
KG_LLM_MODEL = "gemini-2.5-pro-preview-03-25" #"gemini-2.0-flash-exp"
PI_RANKING_LLM_MODEL = "gemini-2.0-flash-thinking-exp-01-21" # Equivalent for 'gemini-2.0-flash-thinking-exp'
ROUTER_LLM_MODEL = "gemini-2.0-flash-lite"  

graph_db = Neo4jGraph(url=neo4j_url, username=neo4j_user, password=neo4j_password, database=neo4j_db)
check_and_populate_graph(graph_db)
kg_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=gemini_api, temperature=0)


GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
NEO4J_URL = os.getenv("NEO4J_CONNECTION_URL")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_SCHOLAR_DB = os.getenv("NEO4J_SCHOLAR")
kg_qa_system = ScholarKGQA(
             google_api_key=GOOGLE_API_KEY,
             neo4j_url=NEO4J_URL,
             neo4j_user=NEO4J_USER,
             neo4j_password=NEO4J_PASSWORD,
             neo4j_db=NEO4J_SCHOLAR_DB,
             llm_model_tag=KG_LLM_MODEL,
             example=example,
             verbose=True
             )
pi_ranking_system = PIRankingAnalysis()
query_router = QueryRouter(kg_system=kg_qa_system, pi_system=pi_ranking_system, router_llm_model_name="gemini-1.5-flash")


# --- 5. Define Main Handler Function for the App ---
# This is the function that app.py will import
def route_question(question: str) -> dict:
    """
    This is the single entry point for the Flask app. It uses the router
    and ensures the output is always a dictionary formatted for the frontend.
    """
    result = query_router.route(question)
    if isinstance(result, str):
        return {'answer': result, 'generated_query': "N/A (Handled by PI System)", 'full_context': "N/A (Handled by PI System)", 'error': None}
    elif isinstance(result, dict):
        return result
    else:
        return {'answer': "An unexpected error occurred in the router.", 'error': "Invalid return type."}

# This is the new final print statement to prove the correct file is running
print("--- All systems (KG, PI, Router) initialized. Ready for queries. ---")