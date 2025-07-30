# combined_data.py (This file should be in the same directory as app.py)

import os
import re
import io
import sys
import json
import time
import warnings
import pandas as pd
import numpy as np
from datetime import datetime
from dotenv import load_dotenv
from langchain_community.graphs import Neo4jGraph
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import GraphCypherQAChain
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import google.generativeai as genai # For PI ranking and Router LLM calls
import textwrap
from typing import List, Dict, Tuple, Optional, Any
import collections
from sklearn.preprocessing import MinMaxScaler

# --- GLOBAL DATA VARIABLES FOR THIS MODULE ---
# These are initialized here and populated by load_all_data() when this module is imported.
scholar_data = None
ranking_data = None
ranking_data_grouped = None
embedder = None # Will be initialized to SentenceTransformer instance


# --- Helper Functions ---
def duplicate_row_check(df):
    """Checks for and returns duplicate rows and their count."""
    duplicate_rows = df[df.duplicated()]
    return len(duplicate_rows), duplicate_rows.index.tolist()

def categorize_venue(venue_name):
    """Categorizes venues into Conferences or Journals based on keywords."""
    conference_keywords = ["conf", "symp", "workshop", "proc", "int. conf", "international conference", "conference"]
    journal_keywords = ["jour", "trans", "rev", "mag", "journal"]
    venue_name_lower = str(venue_name).lower()
    if any(keyword in venue_name_lower for keyword in conference_keywords):
        return "Conference"
    elif any(keyword in venue_name_lower for keyword in journal_keywords):
        return "Journal"
    else:
        return "Other"

def safe_get(dictionary, keys):
    """Safely gets nested dictionary values."""
    for key in keys:
        if isinstance(dictionary, dict):
            dictionary = dictionary.get(key)
        else:
            return None
    return dictionary

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


# --- Your System Classes ---
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

    def query(self, question: str) -> dict:
        """
        Takes a user question, queries the graph, and returns structured results.
        This function will be imported and used by the Flask app.
        """
        try:
            output_buffer = io.StringIO()
            original_stdout = sys.stdout
            sys.stdout = output_buffer
            
            # Explicitly generate the Cypher query first using the prompt
            generated_query = self.cypher_generation_prompt.format(
                schema=self.graph.schema,  # Ensure dynamic schema usage
                question=question,
                example=self.example
            )
            
            input_data = {
                "query": generated_query,
                "question": question
            }

            # Run the chain. Pass examples directly if the chain supports it or format into prompt.
            # result = self.chain.invoke({"query": question, "example": example})
            result = self.chain.invoke(input_data)
            
            sys.stdout = original_stdout
            captured_output = output_buffer.getvalue()
            
            # Extract details from verbose output
            cypher_query = "Could not extract Cypher query."
            full_context = "Could not extract context."
            
            if 'Generated Cypher:' in captured_output:
                cypher_part = captured_output.split('Generated Cypher:')[1]
                cypher_query = self.clean_ansi(cypher_part.split('\n\n')[0])
            
            if 'Full Context:' in captured_output:
                context_part = captured_output.split('Full Context:')[1]
                full_context = self.clean_ansi(context_part.split('\n\n\x1b[0m>')[0])

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

# --- Class for PI Ranking and Analysis ---
class PIRankingAnalysis:
    def __init__(self, ranking_data: pd.DataFrame, grouped_data: pd.DataFrame, llm_model_name: str):
        print(f"Initializing PIRankingAnalysis with model: {llm_model_name}")
        self.df = ranking_data
        self.df_grouped = grouped_data
        self.embedder = embedder # Use global embedder
        try:
            # Use specific Gemini model via genai SDK
            self.model = genai.GenerativeModel(llm_model_name)
            print(f"PI Ranking LLM ({llm_model_name}) initialized.")
        except Exception as e:
            print(f"Error initializing PI Ranking LLM: {e}")
            self.model = None
            
    def _get_llm_response(self, prompt: str) -> Optional[str]:
        """Internal helper to call the PI Ranking LLM."""
        if not self.model: return "Error: PI Ranking LLM not initialized."
        start_time = time.time()
        try:
            response = self.model.generate_content(prompt)
            duration = time.time() - start_time
            print(f"PI Ranking LLM response generated in {duration:.2f} seconds.")
            # Adapt response extraction based on genai SDK version/behavior
            if hasattr(response, 'text'): return response.text
            if hasattr(response, 'candidates') and response.candidates: return response.candidates[0].content.parts[0].text
            return None
        except Exception as e:
            duration = time.time() - start_time
            print(f"Error during PI Ranking LLM call ({duration:.2f}s): {e}")
            return None
        
    # --- PI Recommendation Methods (adapted from functions) ---
    def _format_pi_data_for_prompt(self, filtered_df: pd.DataFrame, pi_ids_to_format: List[str]) -> Tuple[str, Dict[str, str]]:
        # Identical to format_pi_data_for_prompt function above
        formatted_data = ""
        pi_names = {}
        # ... (copy logic from the function version) ...
        if filtered_df.empty:
             formatted_data = "No data could be retrieved for the specified potential collaborators.\n"
             for pi_id in pi_ids_to_format: pi_names[pi_id] = f"PI ID {pi_id}"
             return formatted_data, pi_names
        for pi_id in pi_ids_to_format:
             pi_specific_data = filtered_df[filtered_df['pi_id'] == pi_id]
             if not pi_specific_data.empty:
                 full_name = pi_specific_data['pi_full_name'].iloc[0]
                 department = pi_specific_data['department'].iloc[0]
                 pi_names[pi_id] = full_name
                 formatted_data += f"--- Researcher: {full_name} (ID: {pi_id}) ---\n"
                 formatted_data += f"Department: {department}\n"
                 formatted_data += "Relevant Roles & Awards Found:\n"
                 for _, row in pi_specific_data.iterrows():
                      formatted_data += f"- Role: {row.get('role', 'N/A')}\n"
                      formatted_data += f"  Award Title: {row.get('award_title', 'N/A')}\n"
                      formatted_data += f"  Start Date: {row.get('start_date', 'N/A')}\n"
                      abstract_preview = textwrap.shorten(str(row.get('abstract', 'N/A')), width=150, placeholder="...")
                      formatted_data += f"  Abstract Snippet: {abstract_preview}\n"
                      formatted_data += f"  Program Element/Reference: {row.get('program_element', 'N/A')} / {row.get('program_reference', 'N/A')}\n\n"
             else:
                 formatted_data += f"--- Researcher ID: {pi_id} ---\nNo award data found.\n\n"
                 pi_names[pi_id] = f"PI ID {pi_id}"
        return formatted_data, pi_names

    def _generate_recommendation_prompt(self, formatted_data_string: str, pi_names_dict: Dict[str, str], research_topic: str) -> str:
         # Identical to generate_recommendation_prompt function above
         collaborator_names_list = ", ".join(pi_names_dict.values())
         # ... (copy logic from the function version) ...
         prompt = f"""Context:
            The following researchers ({collaborator_names_list}) are candidates for a new research project focused on '{research_topic}'. Below is information about their past grants and roles:

            {formatted_data_string}

            Task:
            Based ONLY on the information provided above, analyze the qualifications and past work relevance for each researcher ({collaborator_names_list}).
            Recommend which ONE of these individuals would be the MOST suitable Principal Investigator (PI) to lead this project on '{research_topic}'.

            Provide a detailed explanation for your recommendation. Consider:
            - Direct relevance of past research (titles, abstracts, programs) to '{research_topic}'.
            - Demonstrated experience (e.g., number of awards, roles held like 'Principal Investigator').

            Clearly state the recommended PI by name and justify your choice using specific evidence from the provided context. If the data is insufficient, state that clearly.
            """
         return prompt

    def recommend_pi(self, pi_ids: List[str], research_topic: str) -> str:
        if self.df is None: return "Error: Ranking data not loaded."
        pi_ids_str = [str(pid) for pid in pi_ids]
        filtered_data = self.df[self.df['pi_id'].isin(pi_ids_str)].copy()
        if filtered_data.empty: return f"No ranking data found for PI IDs: {pi_ids_str}"

        formatted_text, pi_names = self._format_pi_data_for_prompt(filtered_data, pi_ids_str)
        prompt_text = self._generate_recommendation_prompt(formatted_text, pi_names, research_topic)
        recommendation = self._get_llm_response(prompt_text)
        return recommendation if recommendation else "Could not generate PI recommendation."

    # --- Influencer Methods ---
    def _get_collaborators_for_awards(self, award_titles: List[str]) -> Dict[str, List[str]]:
         # Helper from PI_ranking.ipynb
         # ... (copy logic from function version) ...
         collaborators = {}
         relevant_awards_df = self.df[self.df['award_title'].isin(award_titles)]
         for title in award_titles:
              award_pis = relevant_awards_df[
                   (relevant_awards_df['award_title'] == title) &
                   (relevant_awards_df['role'].isin(['Principal Investigator', 'Co-Principal Investigator']))
              ]
              names = [name for name in award_pis['pi_full_name'].unique() if pd.notna(name)]
              collaborators[title] = names
         return collaborators

    def _format_influencer_data(self, pi_ids: List[str]) -> Tuple[str, Dict[str, str]]:
         # Simplified version - copy from function above
         formatted_data = ""
         pi_names = {}
         # ... (copy logic from function version) ...
         filtered_df = self.df[self.df['pi_id'].isin(pi_ids)].copy()
         if filtered_df.empty: return "No data found for specified PI IDs.", {}
         for pi_id in pi_ids:
             pi_specific_data = filtered_df[filtered_df['pi_id'] == pi_id]
             if not pi_specific_data.empty:
                  full_name = pi_specific_data['pi_full_name'].iloc[0]
                  pi_names[pi_id] = full_name
                  formatted_data += f"--- Potential Influencer: {full_name} (ID: {pi_id}) ---\n"
                  unique_award_titles = pi_specific_data['award_title'].unique()
                  num_projects = len(unique_award_titles)
                  formatted_data += f"Total Projects Involved In: {num_projects}\n"
                  collaborators_by_award = self._get_collaborators_for_awards(list(unique_award_titles))
                  all_collaborators = set(name for names in collaborators_by_award.values() for name in names if name != full_name)
                  num_unique_collaborators = len(all_collaborators)
                  formatted_data += f"Total Unique Collaborators: {num_unique_collaborators}\n"
                  unique_elements = pi_specific_data['program_element'].dropna().unique()
                  unique_references = pi_specific_data['program_reference'].dropna().unique()
                  all_fields = set(unique_elements) | set(unique_references)
                  num_unique_fields = len(all_fields)
                  formatted_data += f"Number of Unique Research Fields: {num_unique_fields}\n\n"
             else:
                  formatted_data += f"--- Potential Influencer ID: {pi_id} ---\nNo data found.\n\n"
                  pi_names[pi_id] = f"PI ID {pi_id}"
         return formatted_data, pi_names


    def _generate_influencer_prompt(self, formatted_data_string: str, pi_names_dict: Dict[str, str]) -> str:
         # Copy from function version above
         candidate_names_list = ", ".join(pi_names_dict.values())
         # ... (copy logic from function version) ...
         prompt = f"""Context:
            You are analyzing research data to identify 'influencers' based on:
            1. Number of distinct projects/awards.
            2. Number of unique collaborators.
            3. Diversity of research fields (Program Elements/References).

            Below is summarized data for potential influencers ({candidate_names_list}):

            {formatted_data_string}

            Task:
            Based ONLY on the summarized information, rank these individuals ({candidate_names_list}) from most influential to least influential according to the criteria above.
            Provide a clear ranking and a concise justification for each, referencing the specific metrics (project count, collaborator count, field count).
            """
         return prompt


    def rank_influencers_by_list(self, pi_ids: List[str]) -> str:
        if self.df is None: return "Error: Ranking data not loaded."
        pi_ids_str = [str(pid) for pid in pi_ids]
        formatted_text, pi_names = self._format_influencer_data(pi_ids_str)
        if not pi_names: return "Could not format data for influencer ranking."
        prompt_text = self._generate_influencer_prompt(formatted_text, pi_names)
        ranking_result = self._get_llm_response(prompt_text)
        return ranking_result if ranking_result else "Could not generate influencer ranking."

    # --- Influencer by Criterion Methods ---
    def _select_candidate_pis(self, criterion_type: str, criterion_value: str, top_k: int = 10) -> List[str]:
         # Copy logic from select_candidate_pis_v2 function above
         if self.df is None or self.df_grouped is None: return []
         # ... (copy logic, using self.df and self.df_grouped) ...
         print(f"Selecting top {top_k} candidates based on {criterion_type}: '{criterion_value}'...")
         candidate_ids = []
         if criterion_type == "topic":
             if 'text_embedding' not in self.df_grouped.columns: return []
             topic_emb = self.embedder.encode(criterion_value)
             valid_embeddings = self.df_grouped['text_embedding'][self.df_grouped['text_embedding'].apply(lambda x: isinstance(x, np.ndarray))]
             if valid_embeddings.empty: return []
             all_embeddings = np.stack(valid_embeddings.values)
             similarities = cosine_similarity([topic_emb], all_embeddings)[0]
             top_indices = np.argsort(similarities)[::-1][:top_k]
             candidate_ids = self.df_grouped.iloc[valid_embeddings.index[top_indices]]['pi_id'].tolist()
         elif criterion_type == "department":
             if 'department' not in self.df.columns: return []
             dept_match_df = self.df[self.df['department'].str.contains(criterion_value, case=False, na=False)]
             if dept_match_df.empty: return []
             unique_dept_pi_ids = dept_match_df['pi_id'].unique()
             if len(unique_dept_pi_ids) > top_k:
                 candidate_subset = self.df_grouped[self.df_grouped['pi_id'].isin(unique_dept_pi_ids)]
                 if 'award_count' in candidate_subset.columns:
                      ranked_candidates = candidate_subset.sort_values(by='award_count', ascending=False)
                      candidate_ids = ranked_candidates.head(top_k)['pi_id'].tolist()
                 else: candidate_ids = list(unique_dept_pi_ids)[:top_k]
                 print(f"  (Found {len(unique_dept_pi_ids)} PIs, selecting top {top_k} based on award count)")
             else: candidate_ids = list(unique_dept_pi_ids)
         else: return []
         print(f"Selected candidate PI IDs: {candidate_ids}")
         return candidate_ids


    def find_influencer_by_criterion(self, criterion_type: str, criterion_value: str, top_k: int = 5) -> str:
        candidate_pi_ids = self._select_candidate_pis(criterion_type, criterion_value, top_k=top_k)
        if not candidate_pi_ids:
            return f"Could not find influencers matching {criterion_type}: '{criterion_value}'."
        # Rank the selected candidates
        return self.rank_influencers_by_list(candidate_pi_ids)

class QueryRouter:
    def __init__(self, kg_system: ScholarKGQA, pi_system: PIRankingAnalysis, router_llm_model_name: str):
        print(f"Initializing QueryRouter with model: {router_llm_model_name}")
        self.kg_system = kg_system
        self.pi_system = pi_system
        try:
            # genai.configure() should be done globally in app.py before this is instantiated
            self.router_model = genai.GenerativeModel(router_llm_model_name)
            print(f"Router LLM ({router_llm_model_name}) initialized.")
        except Exception as e:
            print(f"Error initializing Router LLM: {e}")
            self.router_model = None

    def _get_intent_from_llm(self, question: str) -> str:
        if not self.router_model: return "unknown"
        prompt = f"""Classify the user's question into one of the following categories:
            - kg_query: Asking about papers, authors, venues, citations, specific collaborations, disciplines.
            - pi_recommendation: Asking to recommend a PI for a topic from a given list of researchers.
            - influencer_list: Asking to rank or find the best influencer from a given list of researchers.
            - influencer_topic: Asking to find the best influencer for a specific research topic.
            - influencer_dept: Asking to find the best influencer within a specific department.
            - unknown: If the question doesn't fit the above categories.

            Question: "{question}"
            Category: """
        try:
            safety_settings = [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            ]
            response = self.router_model.generate_content(prompt, safety_settings=safety_settings)
            if hasattr(response, 'text'): intent = response.text.strip().lower()
            elif hasattr(response, 'candidates') and response.candidates: intent = response.candidates[0].content.parts[0].text.strip().lower()
            else: intent = "unknown"

            valid_intents = ["kg_query", "pi_recommendation", "influencer_list", "influencer_topic", "influencer_dept", "unknown"]
            intent = re.sub(r'[^\w_]', '', intent)
            if intent not in valid_intents:
                 print(f"Warning: LLM returned unexpected intent '{intent}'. Defaulting to 'unknown'.")
                 return "unknown"
            print(f"LLM classified intent as: {intent}")
            return intent
        except Exception as e:
            if "block_reason" in str(e):
                 print(f"Intent classification blocked: {e}")
                 return "unknown"
            print(f"Error during intent classification: {e}")
            return "unknown"

    def _extract_parameters(self, question: str, intent: str) -> Dict[str, Any]:
        params = {}
        print(f"Attempting to extract parameters for intent: {intent}")
        if intent == "pi_recommendation":
            id_match = re.search(r"(\[.*?\])", question, re.IGNORECASE)
            pi_ids = []
            if id_match:
                pi_ids_str = id_match.group(1)
                pi_ids = self._extract_pi_ids(pi_ids_str)
                if pi_ids: params["pi_ids"] = pi_ids
                print(f"Found PI IDs: {pi_ids}")

            topic_match = re.search(r"\s(?:in|topic)\s+[\'\"]?([^\'\"]+)[\'\"]?", question, re.IGNORECASE)
            if topic_match:
                topic = topic_match.group(1).strip()
                if not topic.startswith('['):
                     params["topic"] = topic
                     print(f"Found Topic: {topic}")

        elif intent == "influencer_list":
             match = re.search(r"among\s+(?:them|these)\s*:\s*(.*)", question, re.IGNORECASE)
             if match:
                 pi_ids = self._extract_pi_ids(match.group(1))
                 if pi_ids: params["pi_ids"] = pi_ids

        elif intent == "influencer_topic":
             match = re.search(r"topic\s+[\'\"]?([^\'\"]+)[\'\"]?", question, re.IGNORECASE)
             if match:
                 params["topic"] = match.group(1).strip()

        elif intent == "influencer_dept":
             match = re.search(r"in\s+[\'\"]?([^\'\"]+)[\'\"]?\s+department", question, re.IGNORECASE)
             if match:
                 params["department"] = match.group(1).strip()

        print(f"Final extracted parameters: {params}")
        return params

    def _extract_pi_ids(self, text: str) -> List[str]:
         match = re.search(r"\[\s*([\'\"]?\s*\d+\s*[\'\"]?(?:\s*,\s*[\'\"]?\d+\s*[\'\"]?)*)\s*\]", text)
         if match:
             id_list_str = match.group(1)
             ids = [item.strip('\'" ') for item in id_list_str.split(',')]
             return [item for item in ids if item.isdigit()]
         return []

    def route(self, question: str) -> Dict[str, Any]:
        """Routes the question to the correct system and method and returns a dictionary response."""
        print(f"\nRouting question: \"{question}\"")
        intent = self._get_intent_from_llm(question)
        print(f"Identified intent: {intent}")

        params = self._extract_parameters(question, intent)
        print(f"Extracted parameters: {params}")

        answer = "No response generated."
        generated_query = None
        full_context = None
        error = None

        if intent == "pi_recommendation":
            if "pi_ids" in params and params["pi_ids"] and "topic" in params and params["topic"]:
                 answer = self.pi_system.recommend_pi(params["pi_ids"], params["topic"])
            else:
                 print("Warning: PI Recommendation intent identified, but parameters extraction failed. Falling back to KG.")
                 kg_result = self.kg_system.query(question)
                 answer = kg_result.get('answer', 'KG query returned no specific result.')
                 generated_query = kg_result.get('generated_query')
                 full_context = kg_result.get('full_context')

        elif intent == "influencer_list":
             if "pi_ids" in params and params["pi_ids"]:
                 answer = self.pi_system.rank_influencers_by_list(params["pi_ids"])
             else:
                  print("Warning: Influencer List intent identified, but parameters extraction failed. Falling back to KG.")
                  kg_result = self.kg_system.query(question)
                  answer = kg_result.get('answer', 'KG query returned no specific result.')
                  generated_query = kg_result.get('generated_query')
                  full_context = kg_result.get('full_context')

        elif intent == "influencer_topic":
             if "topic" in params and params["topic"]:
                 answer = self.pi_system.find_influencer_by_criterion("topic", params["topic"])
             else:
                  print("Warning: Influencer Topic intent identified, but parameters extraction failed. Falling back to KG.")
                  kg_result = self.kg_system.query(question)
                  answer = kg_result.get('answer', 'KG query returned no specific result.')
                  generated_query = kg_result.get('generated_query')
                  full_context = kg_result.get('full_context')

        elif intent == "influencer_dept":
             if "department" in params and params["department"]:
                 answer = self.pi_system.find_influencer_by_criterion("department", params["department"])
             else:
                  print("Warning: Influencer Dept intent identified, but parameters extraction failed. Falling back to KG.")
                  kg_result = self.kg_system.query(question)
                  answer = kg_result.get('answer', 'KG query returned no specific result.')
                  generated_query = kg_result.get('generated_query')
                  full_context = kg_result.get('full_context')

        elif intent == "kg_query":
            print("Routing to KG system for query...")
            kg_result = self.kg_system.query(question)
            answer = kg_result.get('answer', 'KG query returned no specific result.')
            generated_query = kg_result.get('generated_query')
            full_context = kg_result.get('full_context')
        else: # Fallback for 'unknown' intent
            print("Intent unclear or extraction failed, falling back to Knowledge Graph query.")
            kg_result = self.kg_system.query(question)
            answer = kg_result.get('answer', 'Fallback KG query returned no specific result.')
            generated_query = kg_result.get('generated_query')
            full_context = kg_result.get('full_context')

        return {
            'answer': answer,
            'generated_query': generated_query,
            'full_context': full_context,
            'error': error # Include error if applicable
        }


# --- Data Loading Function (MOVED HERE) ---
def load_all_data(scholar_csv_path="../data/scholer_recommendation.csv", ranking_data_dir="../data/ranking_data/"):
    """
    Loads and preprocesses both datasets.
    It checks for pre-processed Parquet files and loads them if they exist.
    Otherwise, it processes the raw data and saves the processed DataFrames.
    Populates global variables: scholar_data, ranking_data, ranking_data_grouped, embedder.
    """
    global scholar_data, ranking_data, ranking_data_grouped, embedder # Explicitly declare globals

    processed_data_dir = "../data/processed/"
    scholar_processed_path = os.path.join(processed_data_dir, "scholar_data.parquet")
    ranking_processed_path = os.path.join(processed_data_dir, "ranking_data.parquet")
    ranking_grouped_processed_path = os.path.join(processed_data_dir, "ranking_data_grouped.parquet")

    os.makedirs(processed_data_dir, exist_ok=True)

    # --- Try loading pre-processed Scholar Data ---
    print("Attempting to load pre-processed Scholar Data...")
    if os.path.exists(scholar_processed_path):
        try:
            scholar_data = pd.read_parquet(scholar_processed_path)
            print(f"Pre-processed Scholar data loaded from {scholar_processed_path}. Shape: {scholar_data.shape}")
        except Exception as e:
            print(f"Error loading pre-processed Scholar data: {e}. Re-processing from raw CSV...")
            scholar_data = None
    else:
        print("Pre-processed Scholar data not found. Processing from raw CSV...")

    if scholar_data is None: # If not loaded from parquet, process from CSV
        try:
            scholar_data = pd.read_csv(scholar_csv_path)
            scholar_data = scholar_data.drop(columns=["Abstract", "Keywords"], axis=1, errors='ignore')
            scholar_data = pd.concat([scholar_data.head(80), scholar_data.tail(20)], ignore_index=True)
            scholar_data.rename(columns={'Fields of Study': 'Discipline', 'Category': 'Topic'}, inplace=True)
            print("Scholar data loaded from CSV.")
            print(f"Scholar data processing started. {len(scholar_data)} rows found.")
            # Your existing scholar data preprocessing steps
            scholar_data['Authors_list'] = scholar_data['Authors'].str.split(',')
            scholar_data = scholar_data.explode('Authors_list').reset_index(drop=True)
            scholar_data["Authors"] = scholar_data["Authors_list"]
            scholar_data.drop(["Authors_list"], axis=1, inplace=True)

            scholar_data['Discipline_list'] = scholar_data['Discipline'].str.split(',')
            scholar_data = scholar_data.explode('Discipline_list').reset_index(drop=True)
            scholar_data["Discipline"] = scholar_data["Discipline_list"]
            scholar_data.drop(["Discipline_list"], axis=1, inplace=True)
            _, duplicate_list = duplicate_row_check(scholar_data)
            if duplicate_list:
                scholar_data.drop(duplicate_list, inplace=True)
                scholar_data.reset_index(drop=True, inplace=True)

            scholar_data.dropna(inplace=True)
            _, duplicate_list = duplicate_row_check(scholar_data)
            if duplicate_list:
                scholar_data.drop(duplicate_list, inplace=True)
                scholar_data.reset_index(drop=True, inplace=True)

            scholar_data.rename(columns={'Title': 'Paper Title', 'Authors': 'Author', 'Year': 'Year Published'}, inplace=True)
            scholar_data['Venue Type'] = scholar_data['Venue'].apply(categorize_venue)
            scholar_data = scholar_data.applymap(lambda x: x.strip() if isinstance(x, str) else x)
            _, duplicate_list = duplicate_row_check(scholar_data)
            if duplicate_list:
                scholar_data.drop(duplicate_list, inplace=True)
                scholar_data.reset_index(drop=True, inplace=True)

            print(f"Scholar data processing complete. {len(scholar_data)} rows loaded.")
            scholar_data.to_parquet(scholar_processed_path, index=False)
            print(f"Processed Scholar data saved to {scholar_processed_path}")
        except Exception as e:
            print(f"Error processing raw Scholar data: {e}")
            scholar_data = None


    # --- Try loading pre-processed Ranking Data ---
    print("\nAttempting to load pre-processed Ranking Data...")
    if os.path.exists(ranking_processed_path) and os.path.exists(ranking_grouped_processed_path):
        try:
            ranking_data = pd.read_parquet(ranking_processed_path)
            ranking_data_grouped = pd.read_parquet(ranking_grouped_processed_path)
            embedder = SentenceTransformer("all-MiniLM-L6-v2") # Re-initialize embedder
            print(f"Pre-processed Ranking data loaded from {ranking_processed_path} and {ranking_grouped_processed_path}.")
            print(f"Ranking data shape: {ranking_data.shape}, Grouped data shape: {ranking_data_grouped.shape}")
        except Exception as e:
            print(f"Error loading pre-processed Ranking data: {e}. Re-processing from raw JSON files...")
            ranking_data = None
            ranking_data_grouped = None
    else:
        print("Pre-processed Ranking data not found. Processing from raw JSON files...")

    if ranking_data is None or ranking_data_grouped is None:
        print(f"Loading Ranking Data from raw JSONs...")
        records = []
        if not os.path.exists(ranking_data_dir):
            print(f"Error: Ranking data directory not found at {ranking_data_dir}")
        else:
            for sub_dir in os.listdir(ranking_data_dir):
                sub_directory = os.path.join(ranking_data_dir, sub_dir)
                if os.path.isdir(sub_directory):
                    print(f"Reading files in {sub_dir}...")
                    for filename in os.listdir(sub_directory):
                        if filename.endswith('.json'):
                            filepath = os.path.join(sub_directory, filename)
                            try:
                                with open(filepath, 'r') as file:
                                    data = json.load(file)
                                award_type = data.get("awd_istr_txt")
                                award_title = data.get("awd_titl_txt")
                                abstract = data.get("abst_narr_txt")
                                org_name = data.get("org_long_name")
                                org_name2 = data.get("org_long_name2")
                                perf_inst_name = safe_get(data, ["perf_inst", "perf_inst_name"])

                                pgm_ele_list = data.get("pgm_ele")
                                program_element = pgm_ele_list[0].get("pgm_ele_long_name") if isinstance(pgm_ele_list, list) and len(pgm_ele_list) > 0 else None

                                pgm_ref_list = data.get("pgm_ref")
                                program_reference = pgm_ref_list[0].get("pgm_ref_long_name") if isinstance(pgm_ref_list, list) and len(pgm_ref_list) > 0 else None

                                pi_list = data.get("pi")
                                if not isinstance(pi_list, list):
                                    continue

                                for pi in pi_list:
                                    record = {
                                        "award_type": award_type,
                                        "award_title": award_title,
                                        "abstract": abstract,
                                        "org_name": org_name,
                                        "org_name2": org_name2,
                                        "perf_inst_name": perf_inst_name,
                                        "program_element": program_element,
                                        "program_reference": program_reference,
                                        "pi_id": pi.get("pi_id"),
                                        "pi_full_name": pi.get("pi_full_name", "").strip() if pi.get("pi_full_name") else None,
                                        "role": pi.get("proj_role_code2", "").strip() if pi.get("proj_role_code2") else None,
                                        "department": pi.get("pi_dept_name"),
                                        "email": pi.get("pi_email_addr"),
                                        "start_date": pi.get("start_date")
                                    }
                                    records.append(record)
                            except Exception as e:
                                print(f"Error reading or processing {filepath}: {e}")
                                continue

        ranking_data = pd.DataFrame(records)
        if not ranking_data.empty:
            ranking_data = ranking_data[ranking_data['role'].isin(['Co-Principal Investigator', 'Principal Investigator'])].copy()
            print(f"Ranking data loaded and filtered. {len(ranking_data)} rows found.")

            print(f"Preparing ranking data group for analysis...")
            text_columns = [
                "award_type", "award_title", "abstract",
                "org_name", "org_name2", "perf_inst_name",
                "program_element", "program_reference"
            ]
            ranking_data["combined_text"] = ranking_data[text_columns].fillna('').astype(str).agg(" ".join, axis=1)

            ranking_data["leadership"] = ranking_data["role"].apply(lambda x: 1 if "Principal Investigator" in str(x) else 0)

            ranking_data["start_date"] = pd.to_datetime(ranking_data["start_date"], errors='coerce')
            reference_date = datetime.now()
            ranking_data["experience_years"] = (reference_date - ranking_data["start_date"]).dt.days / 365.25
            ranking_data.dropna(subset=['experience_years'], inplace=True)

            embedder = SentenceTransformer("all-MiniLM-L6-v2")
            ranking_data["text_embedding"] = ranking_data["combined_text"].fillna('').apply(lambda x: embedder.encode(x))

            award_counts = ranking_data.groupby("pi_id").size().reset_index(name="award_count")
            ranking_data_grouped = ranking_data.groupby("pi_id").agg(
                experience_years=("experience_years", "mean"),
                leadership=("leadership", "max"),
                text_embedding=("text_embedding", lambda embs: np.mean(np.stack(embs), axis=0))
            ).reset_index()
            ranking_data_grouped = ranking_data_grouped.merge(award_counts, on="pi_id", how="left")

            scaler = MinMaxScaler()
            ranking_data_grouped[["exp_norm", "award_norm"]] = scaler.fit_transform(ranking_data_grouped[["experience_years", "award_count"]])

            ranking_data.to_parquet(ranking_processed_path, index=False)
            ranking_data_grouped.to_parquet(ranking_grouped_processed_path, index=False)
            print(f"Processed Ranking data saved to {ranking_processed_path} and {ranking_grouped_processed_path}")
        else:
            print("No ranking data could be processed from raw JSONs.")
            ranking_data = pd.DataFrame()
            ranking_data_grouped = pd.DataFrame()
            embedder = SentenceTransformer("all-MiniLM-L6-v2")

    print(f"Final data shapes: Ranking: {ranking_data.shape}, Grouped Ranking: {ranking_data_grouped.shape}")
    
    # CALL load_all_data ONCE WHEN THIS MODULE IS IMPORTED
# This ensures the global variables (scholar_data, ranking_data, etc.) are populated
# before any other module (like app.py) tries to import them.
load_all_data()
