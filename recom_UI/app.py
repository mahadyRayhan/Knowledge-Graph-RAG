# app.py

# Removed MinMaxScaler, it's used in combined_data.py
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
import os
import google.generativeai as genai

# Import all necessary components and GLOBAL data variables from combined_data.py
# The load_all_data function will be called during the import process of combined_data.py
# if it's placed at the module level in combined_data.py.
# Otherwise, we'll explicitly call it in initialize_systems.
from combined_data import (
    ScholarKGQA,
    PIRankingAnalysis,
    QueryRouter,
    load_all_data, # Explicitly import the function
    # These are the GLOBAL variables that load_all_data in combined_data.py will populate.
    # We import them here so app.py refers to the same objects.
    scholar_data,
    ranking_data,
    ranking_data_grouped,
    embedder
)

app = Flask(__name__)

# --- Configuration & Setup ---
load_dotenv(dotenv_path="../.env")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
NEO4J_URL = os.getenv("NEO4J_CONNECTION_URL")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_SCHOLAR_DB = os.getenv("NEO4J_SCHOLAR")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found. Please set it in your .env file.")
if not all([NEO4J_URL, NEO4J_USER, NEO4J_PASSWORD, NEO4J_SCHOLAR_DB]):
     raise ValueError("Neo4j details not fully found. Please check your .env file.")

# Configure the Gemini API globally for the application
try:
    genai.configure(api_key=GOOGLE_API_KEY)
    print("Gemini API key configured successfully.")
except Exception as e:
    print(f"Error configuring Gemini API key: {e}")


# --- Constants ---
KG_LLM_MODEL = "gemini-2.5-pro-preview-03-25"
PI_RANKING_LLM_MODEL = "gemini-2.0-flash-thinking-exp-01-21"
ROUTER_LLM_MODEL = "gemini-2.0-flash-lite"

# --- Global System Instances (for app.py's state) ---
kg_system_instance = None
pi_system_instance = None
router_instance = None
system_initialized = False

def initialize_systems():
    global kg_system_instance, pi_system_instance, router_instance, system_initialized
    # No need to declare scholar_data, ranking_data, etc. as global here.
    # They are already imported globals from combined_data.py.

    if system_initialized:
        print("Systems already initialized.")
        return

    print("--- System Initialization ---")
    try:
        # Explicitly call load_all_data from combined_data.py
        # This populates the global variables in combined_data.py,
        # which app.py then accesses via import.
        load_all_data()

        # IMPORTANT: Check if data was successfully loaded/processed from combined_data.py
        # These are the variables imported from combined_data.py
        
        if scholar_data is None:
            print("scholar_data is None")
        else:
            print("scholar_data exists")

        if ranking_data is None:
            print("ranking_data is None")
        else:
            print("ranking_data exists")

        if ranking_data_grouped is None:
            print("ranking_data_grouped is None")
        else:
            print("ranking_data_grouped exists")

        
        if scholar_data is None or ranking_data is None or ranking_data_grouped is None:
            print("Critical Error: Data loading/processing failed in combined_data.py. Cannot initialize systems.")
            system_initialized = False
            return # Exit the function if data is not ready

        # Step 3: Initialize PI Ranking System
        pi_system_instance = PIRankingAnalysis(
            ranking_data=ranking_data,
            grouped_data=ranking_data_grouped,
            llm_model_name=PI_RANKING_LLM_MODEL
        )
        print("PI Ranking System initialized.")

        # Step 4: Initialize Scholar KG System
        scholar_kg_system = ScholarKGQA(
            google_api_key=GOOGLE_API_KEY,
            neo4j_url=NEO4J_URL,
            neo4j_user=NEO4J_USER,
            neo4j_password=NEO4J_PASSWORD,
            neo4j_db=NEO4J_SCHOLAR_DB,
            llm_model_tag=KG_LLM_MODEL,
            example=None,
            verbose=True
        )
        print("Scholar KG System initialized.")

        # Step 5: Initialize Query Router
        router_instance = QueryRouter(
            scholar_kg_system, pi_system_instance, ROUTER_LLM_MODEL
        )
        print("Query Router initialized.")

        # Final check if critical components (like LLM chains) initialized correctly
        if not scholar_kg_system.chain or router_instance.router_model is None:
            print("Critical Error: One or more core system components (e.g., LLM models) failed to initialize.")
            system_initialized = False
        else:
            system_initialized = True
            print("\n--- Systems Ready ---")

    except Exception as e:
        print(f"Failed to initialize all systems due to an unexpected error: {e}")
        system_initialized = False

# Initialize systems when the Flask application starts
with app.app_context():
    initialize_systems()


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/health')
def health_check():
    if system_initialized:
        return jsonify({"status": "healthy", "message": "All systems are operational."}), 200
    else:
        return jsonify({"status": "error", "message": "System components not initialized or encountered an error."}), 500

@app.route('/query', methods=['POST'])
def handle_query():
    if not system_initialized:
        return jsonify({"error": "Backend systems are not ready."}), 503

    data = request.get_json()
    question = data.get('question')

    if not question:
        return jsonify({"error": "No question provided."}), 400

    print(f"Received query: {question}")
    try:
        result = router_instance.route(question)

        answer = result.get('answer', 'No answer generated.')
        generated_query = result.get('generated_query', 'N/A')
        full_context = result.get('full_context', 'N/A')

        print("------------------------------------------")
        print("Result keys:", result.keys())
        print(f"Answer: {answer}")
        print(f"Generated Query: {generated_query}")
        print(f"Full Context: {full_context}")
        print(f"Sending answer: {answer}")
        print("------------------------------------------")

        return jsonify({"answer": answer}), 200
    except Exception as e:
        print(f"Error processing query: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port = 5004)