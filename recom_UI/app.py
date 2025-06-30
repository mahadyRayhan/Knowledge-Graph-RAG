# app.py

import os
import json
import threading
from flask import Flask, request, jsonify, render_template
from graph_setup import query_graph

# Initialize the Flask app
app = Flask(__name__)

# --- History File Setup ---
HISTORY_FILE = 'qna_history.json'
# A lock to prevent race conditions when reading/writing the history file
history_lock = threading.Lock()

# Helper to read history safely
def read_history():
    with history_lock:
        if not os.path.exists(HISTORY_FILE):
            return []
        try:
            with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return []

# Helper to write history safely
def write_history(data):
    with history_lock:
        with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)

# --- Routes ---

@app.route("/")
def home():
    """Renders the main page with the chat interface."""
    return render_template("index.html")

@app.route("/history", methods=['GET'])
def get_history():
    """Endpoint to fetch the entire Q&A history."""
    history = read_history()
    return jsonify(history)

@app.route("/ask", methods=["POST"])
def ask():
    """
    Handles POST requests, gets an answer, and saves the Q&A to the history file.
    """
    if not request.form or "question" not in request.form:
        return jsonify({"error": "Question not provided in form data."}), 400
        
    question = request.form["question"]
    
    if not question:
        return jsonify({"error": "Question cannot be empty."}), 400

    print(f"Received question: {question}")
    
    try:
        results = query_graph(question)
        
        if results.get('error'):
             return jsonify({"error": results['error']}), 500

        explanation_text = "The answer was generated using a multi-step process: ..." # Your explanation text here

        response = {
            "question": question,
            "answer": results['answer'],
            "generated_query": results['generated_query'],
            "full_context": results['full_context'],
            "explanation": explanation_text
        }
        
        # --- Save to history file ---
        history = read_history()
        history.insert(0, response)  # Add new entry to the top of the list
        write_history(history)
        
        return jsonify(response)
        
    except Exception as e:
        print(f"An unexpected error occurred in /ask endpoint: {str(e)}")
        return jsonify({"error": "An internal server error occurred."}), 500

if __name__ == "__main__":
    # The setup in graph_setup.py runs once when this script is started
    app.run(debug=True, port=5004)