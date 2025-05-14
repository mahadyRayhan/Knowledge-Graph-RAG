# -*- coding: utf-8 -*-
import pandas as pd
import json
import os
import re
import time
import datetime
import sys
import numpy as np
from dotenv import load_dotenv
from langchain_community.graphs import Neo4jGraph
from langchain.chains import GraphCypherQAChain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
import traceback # Import traceback for better error logging

# --- Configuration ---
load_dotenv(dotenv_path=".env")
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
NEO4J_URL = os.getenv("NEO4J_CONNECTION_URL")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DB = os.getenv("NEO4J_MOVIE") # Ensure this matches your .env variable for movies
RATINGS_CSV_PATH = "data/ratings.csv"
MOVIES_CSV_PATH = "data/movies.csv"
TAGS_CSV_PATH = "data/tags.csv" # Optional: Path for tags data

RESULTS_FILE = f"movie_performance_test_results.txt"

# --- Testing Parameters ---
NUM_RUNS = 5  # Set 'n' - number of times to run the test for each data size
# Define data sizes - this will be used as `nrows` when reading CSVs
DATA_SIZES = [100, 400, 1000, 2000, 3000, 5000, 10000] # Example sizes, reduced for testing
DATA_SIZES = [50000, 80000]
print(f"Data sizes to test: {DATA_SIZES}")

# --- Helper Functions ---
def separate_title_and_year(title):
    if not isinstance(title, str):
        return title, None
    year_match = re.search(r'\s*\((\d{4})\)\s*$', title)
    year = None
    cleaned_title = title
    if year_match:
        try:
            year = int(year_match.group(1))
            cleaned_title = title[:year_match.start()].strip()
        except (ValueError, IndexError):
             year = None
             cleaned_title = title
    if not cleaned_title:
        cleaned_title = "Unknown Title"
    return cleaned_title, year

# --- Example Set (Required for Cypher Prompt Formatting) ---
# This full list IS STILL NEEDED for the cypher_generation_prompt's {example} field
example = [
    {"question": "Give me 10 movies similar to 'Forrest Gump' released in 1990 or later.","query": "MATCH (m:Movie {title: 'Forrest Gump'})-[:HAS_GENRE]->(g:Genre)<-[:HAS_GENRE] -(similar:Movie) WHERE m <> similar AND similar.year >= 1990 RETURN similar.title AS SimilarMovies, COUNT(g) AS SharedGenres ORDER BY SharedGenres DESC LIMIT 10",},
    {"question": "What are the ratings for 'Forrest Gump'?", "query": "MATCH (m:Movie {title: 'Forrest Gump'})<-[r:RATED]-(u:User) RETURN u.userId AS UserId, r.rating AS Rating",},
    {"question": "What is the average rating for 'Forrest Gump'?", "query": "MATCH (m:Movie {title: 'Forrest Gump'})<-[r:RATED]-(u:User) RETURN AVG(r.rating) AS AverageRating",},
    {"question": "Find all movies released in 1995.", "query": "MATCH (m:Movie {year: 1995}) RETURN m.title AS MoviesReleasedIn1995",},
    {"question": "What are the genres of 'Pocahontas'?", "query": "MATCH (m:Movie {title: 'Pocahontas'})-[:HAS_GENRE]->(g:Genre) RETURN g.name AS Genres",},
    {"question": "How many movies are in the 'Comedy' genre?", "query": "MATCH (g:Genre {name: 'Comedy'})<-[:HAS_GENRE]-(m:Movie) RETURN COUNT(DISTINCT m) AS ComedyMovieCount",},
    {"question": "List all users who rated movies in the 'Drama' genre.", "query": "MATCH (g:Genre {name: 'Drama'})<-[:HAS_GENRE]-(m:Movie)<-[:RATED]-(u:User) RETURN DISTINCT u.userId AS UsersWhoRatedDrama LIMIT 20",},
    {"question": "Find movies with 'City' in the title.", "query": "MATCH (m:Movie) WHERE m.title CONTAINS 'City' RETURN m.title AS MoviesWithCityInTitle",},
    {"question": "What are the ratings for 'Eat Drink Man Woman (Yin shi nan nu)'?", "query": "MATCH (m:Movie {title: 'Eat Drink Man Woman (Yin shi nan nu)'})<-[r:RATED]-(u:User) RETURN u.userId AS UserId, r.rating AS Rating LIMIT 20",},
    {"question": "Which movies have the most shared genres with 'While You Were Sleeping'?", "query": "MATCH (m:Movie {title: 'While You Were Sleeping'})-[:HAS_GENRE]->(g:Genre)<-[:HAS_GENRE]-(similar:Movie) WHERE m <> similar RETURN similar.title AS SimilarMovies, COUNT(g) AS SharedGenres ORDER BY SharedGenres DESC LIMIT 10",},
    {"question": "What is the highest rated movie?", "query": "MATCH (m:Movie)<-[r:RATED]-(u:User) WITH m, AVG(r.rating) AS AverageRating RETURN m.title AS HighestRatedMovie ORDER BY AverageRating DESC LIMIT 1",},
    {"question": "Find all movies that are both 'Comedy' and 'Romance'.", "query": "MATCH (m:Movie)-[:HAS_GENRE]->(g1:Genre {name:'Comedy'}), (m)-[:HAS_GENRE]->(g2:Genre {name:'Romance'}) RETURN m.title AS ComedyRomanceMovies LIMIT 20",},
    {"question": "How many users rated 'French Kiss'?", "query": "MATCH (m:Movie {title: 'French Kiss'})<-[:RATED]-(u:User) RETURN COUNT(DISTINCT u) AS NumberOfUsersWhoRatedFrenchKiss",},
    {"question": "What movies did user 1 rate?", "query": "MATCH (u:User {userId: 1})-[r:RATED]->(m:Movie) RETURN m.title AS MoviesRatedByUser1",},
    {"question": "Find all movies in the 'Action' genre.", "query": "MATCH (m:Movie)-[:HAS_GENRE]->(g:Genre {name: 'Action'}) RETURN m.title AS ActionMovies LIMIT 20",},
    {"question": "List users who rated movies released in 1995.", "query": "MATCH (m:Movie {year: 1995})<-[r:RATED]-(u:User) RETURN DISTINCT u.userId AS UsersWhoRatedMoviesIn1995 LIMIT 20",},
    {"question": "How many movies has each user rated?", "query": "MATCH (u:User)-[r:RATED]->(m:Movie) RETURN u.userId AS UserId, COUNT(m) AS RatedMoviesCount LIMIT 10",},
    {"question": "What are the genres of 'Clueless'?", "query": "MATCH (m:Movie {title: 'Clueless'})-[:HAS_GENRE]->(g:Genre) RETURN g.name AS Genres",},
    {"question": "I like the movie 'True Lies'. give me 5 movie similar to this one with their genres", "query": "MATCH (m:Movie {title: 'True Lies'})-[:HAS_GENRE]->(g:Genre) WITH m, COLLECT(DISTINCT g) AS genres MATCH (similar:Movie)-[:HAS_GENRE]->(sg:Genre) WHERE similar <> m AND sg IN genres WITH similar, COUNT(DISTINCT sg) AS sharedGenreCount, COLLECT(DISTINCT sg.name) as sharedGenres ORDER BY sharedGenreCount DESC LIMIT 5 RETURN similar.title as SimilarMovie, sharedGenres",},
    {"question": "I liked the movie 'Braveheart'. What are 5 similar movies I should watch?", "query": "MATCH (m:Movie {title: 'Braveheart'})-[:HAS_GENRE]->(g:Genre) WITH m, COLLECT(DISTINCT g) AS genres MATCH (similar:Movie)-[:HAS_GENRE]->(sg:Genre) WHERE similar <> m AND sg IN genres WITH similar, COUNT(DISTINCT sg) AS sharedGenreCount ORDER BY sharedGenreCount DESC LIMIT 5 RETURN similar.title as SimilarMovie",},
    {"question": "I like 'Drama' and 'Romance'. I have watched 'Sense and Sensibility' and 'Leaving Las Vegas'. What should I watch next?", "query": "MATCH (g1:Genre {name: 'Drama'}), (g2:Genre {name: 'Romance'}) MATCH (m:Movie)-[:HAS_GENRE]->(g) WHERE g IN [g1, g2] AND NOT m.title IN ['Sense and Sensibility', 'Leaving Las Vegas'] WITH m, COUNT(DISTINCT g) as genreMatchCount WHERE genreMatchCount = 2 RETURN DISTINCT m.title AS RecommendedMovie LIMIT 5",},
    # Add the Jungle Book question here IF you want it used as context for cypher generation LLM
    {"question": "How many users rated 'Jungle Book, The'?", "query": "MATCH (m:Movie {title: 'Jungle Book, The'})<-[:RATED]-(u:User) RETURN COUNT(DISTINCT u) AS NumberOfUsersWhoRatedJungleBook"},
]


# --- Define the specific test question(s) to run ---
TEST_SET_QUESTIONS = ["How many users rated 'Jungle Book, The'?"]


# Define the Cypher query prompt template (needed globally for the test function)
cypher_generation_prompt = PromptTemplate(
     template="""Based on the schema, write a Cypher query to answer the question.

    Schema:
    {schema}

    The question may ask about:
    - Movies with specific genres or titles.
    - Users who rated certain movies or genres.
    - Ratings for movies.
    - Recommendations based on similar genres or watched movies.

    Example questions and queries:
    {example}

    Instructions:
    1. Carefully analyze the question to understand the entities (Movies, Genres, Users) and relationships (HAS_GENRE, RATED) involved.
    2. Use the exact movie titles or genre names provided in the question for filtering (e.g., `m.title = 'Forrest Gump'`, `g.name = 'Comedy'`). Pay attention to case sensitivity if applicable in your data. Use parameters for values.
    3. For similarity questions based on genres, find movies sharing genres with the target movie but exclude the target movie itself. Use `COUNT` and `ORDER BY ... DESC` to find the most similar.
    4. For recommendations based on watched movies and preferred genres, identify genres of the watched movies, find other movies with those genres, and exclude the already watched ones.
    5. Use `DISTINCT` when needed to avoid duplicate results (e.g., listing users or movies).
    6. Use aggregate functions like `AVG`, `COUNT` correctly. For movie average ratings, aggregate per movie (`WITH m, AVG(r.rating) AS avgRating`). For counts of users/movies, use `COUNT(DISTINCT u)` or `COUNT(DISTINCT m)`.
    7. Ensure property names (`m.title`, `m.year`, `m.movieId`, `g.name`, `u.userId`, `r.rating`) match the graph schema.
    8. Add `LIMIT` clauses where appropriate, especially for recommendation or similarity questions, to keep results manageable.

    Question: {question}
    Cypher Query:""",
     input_variables=["schema", "question", "example"],
     partial_variables={"example": str(example)} # Example list IS used by the chain here
)

# --- Main Test Function ---
def run_performance_test(num_rows_to_use, graph_obj, chain_obj, questions_list):
    """
    Loads data using nrows, runs data prep, Neo4j population, and querying.
    Returns execution time in seconds.
    """
    global cypher_generation_prompt, example # Ensure global prompt/example are accessible

    start_time = time.perf_counter()

    # 1. Load specified number of rows directly using nrows
    print(f"    Loading {num_rows_to_use} rows using nrows from CSVs...")
    try:
        ratings_sample = pd.read_csv(RATINGS_CSV_PATH, usecols=['userId', 'movieId', 'rating'], nrows=num_rows_to_use)
        movies_sample = pd.read_csv(MOVIES_CSV_PATH, usecols=['movieId', 'title', 'genres'], nrows=num_rows_to_use)
        # print(f"    Loaded {len(ratings_sample)} ratings rows and {len(movies_sample)} movies rows.")
    except FileNotFoundError as e:
        print(f"    ERROR: CSV file not found during loading: {e}", file=sys.stderr)
        raise e
    except Exception as e:
        print(f"    ERROR loading CSV files: {e}", file=sys.stderr)
        raise e

    if ratings_sample.empty or movies_sample.empty:
        print("    Warning: Loaded data is empty. Skipping further processing.", file=sys.stderr)
        return time.perf_counter() - start_time

    print(f"    Preprocessing loaded data...")
    # 2. Data Preprocessing
    movies_sample[['title', 'year']] = pd.DataFrame(movies_sample['title'].apply(separate_title_and_year).tolist(), index=movies_sample.index)
    movies_sample.dropna(subset=['year'], inplace=True)
    movies_sample['year'] = movies_sample['year'].astype(int)
    df = pd.merge(ratings_sample, movies_sample, on='movieId', how='inner')
    print(f"    Data merged. Shape after merge: {df.shape}") # Reduce verbosity
    if 'genres' in df.columns:
        df['genres_list'] = df['genres'].astype(str).str.split('|')
        movies_exploded = df.explode('genres_list').reset_index(drop=True)
        movies_exploded['genres'] = movies_exploded['genres_list'].str.strip()
        movies_exploded = movies_exploded[movies_exploded['genres'].notna() & (movies_exploded['genres'] != '') & (movies_exploded['genres'] != '(no genres listed)')]
        movies_exploded.drop(["genres_list"], axis=1, inplace=True, errors='ignore')
    else:
        movies_exploded = df
    essential_cols = ['userId', 'movieId', 'rating', 'title', 'year']
    if 'genres' in movies_exploded.columns: essential_cols.append('genres')
    cols_to_check_na = [col for col in essential_cols if col in movies_exploded.columns]
    movies_exploded.dropna(subset=cols_to_check_na, inplace=True)
    movies_exploded.reset_index(drop=True, inplace=True)
    final_rows = len(movies_exploded)
    if final_rows == 0:
        print("    Warning: Preprocessing resulted in zero rows suitable for Neo4j.", file=sys.stderr)
        return time.perf_counter() - start_time
    print(f"    Preprocessing done. {final_rows} relationship entries ready for Neo4j.")

    # 3. Neo4j Interaction
    try:
        graph_obj.query("MATCH (n) DETACH DELETE n")
    except Exception as e:
        print(f"    ERROR clearing Neo4j: {e}.", file=sys.stderr)
    try:
        # Use UNWIND for potentially better performance
        unique_movies = movies_exploded[['movieId', 'title', 'year']].drop_duplicates().copy()
        unique_movies['movieId'] = unique_movies['movieId'].astype(int)
        unique_movies['title'] = unique_movies['title'].astype(str)
        unique_movies['year'] = unique_movies['year'].astype(int)
        movies_list = unique_movies.to_dict('records')
        graph_obj.query("UNWIND $movies as movie_data MERGE (m:Movie {movieId: movie_data.movieId}) ON CREATE SET m.title = movie_data.title, m.year = movie_data.year ON MATCH SET m.title = movie_data.title, m.year = movie_data.year", {'movies': movies_list})

        if 'genres' in movies_exploded.columns:
            unique_genres_relations = movies_exploded[['movieId', 'genres']].drop_duplicates().copy()
            unique_genres_relations['movieId'] = unique_genres_relations['movieId'].astype(int)
            unique_genres_relations['genres'] = unique_genres_relations['genres'].astype(str)
            unique_genre_names = unique_genres_relations[['genres']].drop_duplicates().rename(columns={'genres':'name'})
            genre_list = unique_genre_names.to_dict('records')
            graph_obj.query("UNWIND $genres as genre_data MERGE (g:Genre {name: genre_data.name})", {'genres': genre_list})
            relation_list = unique_genres_relations.to_dict('records')
            graph_obj.query("UNWIND $relations as rel_data MATCH (m:Movie {movieId: rel_data.movieId}) MATCH (g:Genre {name: rel_data.genres}) MERGE (m)-[:HAS_GENRE]->(g)", {'relations': relation_list})

        unique_ratings_relations = movies_exploded[['userId', 'movieId', 'rating']].drop_duplicates().copy()
        unique_ratings_relations['userId'] = unique_ratings_relations['userId'].astype(int)
        unique_ratings_relations['movieId'] = unique_ratings_relations['movieId'].astype(int)
        unique_ratings_relations['rating'] = unique_ratings_relations['rating'].astype(float)
        unique_user_ids = unique_ratings_relations[['userId']].drop_duplicates()
        user_list = unique_user_ids.to_dict('records')
        graph_obj.query("UNWIND $users as user_data MERGE (u:User {userId: user_data.userId})", {'users': user_list})
        rating_relation_list = unique_ratings_relations.to_dict('records')
        graph_obj.query("UNWIND $relations as rel_data MATCH (u:User {userId: rel_data.userId}) MATCH (m:Movie {movieId: rel_data.movieId}) MERGE (u)-[:RATED {rating: rel_data.rating}]->(m)", {'relations': rating_relation_list})
    except Exception as e:
        print(f"    ERROR during Neo4j population phase: {e}", file=sys.stderr)
        raise e
    try:
        graph_obj.refresh_schema()
    except Exception as e:
        print(f"    WARNING: Failed to refresh graph schema: {e}", file=sys.stderr)

    # 4. Run Test Queries via LangChain Chain
    num_questions = len(questions_list)
    print(f"    Running {num_questions} test queries via LangChain...")
    query_errors = 0
    for i, question in enumerate(questions_list):
        try:
            # --- MODIFIED: Use the user-confirmed two-step execution pattern ---

            # Step 1: Format the Cypher generation prompt string
            generated_query_prompt = cypher_generation_prompt.format(
                schema=graph_obj.schema,  # Get schema from the graph object
                question=question,
                example=str(example) # Pass example list as string representation
            )

            # Step 2: Run the chain with the formatted prompt string and original question
            response = chain_obj.run(query=generated_query_prompt, question=question)
            print("" + "-" * 60)
            print(f"Ans: {response}")
            print("" + "-" * 60)

        except Exception as e:
            query_errors += 1
            print(f"      ERROR running query for question #{i+1} ('{question[:50]}...'): {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr) # Keep detailed error logging

    if query_errors > 0:
        print(f"    WARNING: {query_errors}/{num_questions} queries failed during execution.", file=sys.stderr)

    end_time = time.perf_counter()
    return end_time - start_time


# --- Script Execution ---
if __name__ == "__main__":
    # --- Validate Environment Variables ---
    if not all([GEMINI_API_KEY, NEO4J_URL, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DB]):
        print("Error: Missing environment variables.")
        sys.exit(1)

    # --- Initialize Neo4j and LangChain components ONCE ---
    print("Initializing Neo4j connection...")
    try:
        graph = Neo4jGraph(NEO4J_URL, NEO4J_USER, NEO4J_PASSWORD, database=NEO4J_DB)
        graph.query("RETURN 1")
        print(f"Neo4j connection to database '{NEO4J_DB}' successful.")
    except Exception as e:
        print(f"Error initializing Neo4j: {e}")
        sys.exit(1)

    print("Initializing LLM and Chain...")
    try:
        # Define QA prompt (needed for chain initialization)
        qa_prompt = PromptTemplate(
             template="""You are an AI assistant answering questions based on movie graph data.
            Use the provided Cypher query results to answer the question clearly and concisely.

            Question: {question}
            Cypher Query Results: {context}

            Based *only* on the provided results:
            - If the results list movies, genres, or users, present them in a readable list format.
            - If the results contain a number (e.g., count, average rating), state it clearly.
            - If the results are empty or indicate no data found, explicitly state that (e.g., "No movies were found matching that criteria," "There are no ratings available for that movie in the dataset," "No similar movies could be recommended based on the available data.").
            - Do not add any information not present in the results.
            - Make the answer sound natural and helpful.

            Answer:""",
             input_variables=["question", "context"],
        )

        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GEMINI_API_KEY, temperature=0)

        # Note: cypher_generation_prompt is defined globally now
        chain = GraphCypherQAChain.from_llm(
            llm=llm,
            graph=graph,
            verbose=False,
            cypher_generation_prompt=cypher_generation_prompt, # Provide generation prompt
            qa_prompt=qa_prompt, # Provide QA prompt
            validate_cypher=True,
            allow_dangerous_requests=True
        )
        print("LLM and Chain initialized.")
    except Exception as e:
        print(f"Error initializing LangChain components: {e}")
        sys.exit(1)

    # --- Run the timing experiment ---
    all_results_data = []
    print(f"\nStarting movie performance test suite (using nrows, user-specified run method)...")
    print(f"Number of runs per data size (n): {NUM_RUNS}")
    print(f"Data sizes (nrows) to test: {DATA_SIZES}")
    print(f"Test Question(s): {TEST_SET_QUESTIONS}")
    print(f"Results will be saved to: {RESULTS_FILE}")
    print("-" * 60)

    # Main test loop
    for size in DATA_SIZES:
        print(f"\n--- Testing with data size (nrows): {size} ---")
        current_size_times = []
        for i in range(NUM_RUNS):
            print(f"  Starting run {i+1}/{NUM_RUNS} for size {size}...")
            try:
                duration = run_performance_test(size, graph, chain, TEST_SET_QUESTIONS)
                current_size_times.append(duration)
                print(f"  Run {i+1} finished. Duration: {duration:.4f} seconds")
            except Exception as e:
                print(f"  Run {i+1} FAILED: {e}", file=sys.stderr)
                traceback.print_exc(file=sys.stderr)
                current_size_times.append(None)

        # Calculate average
        valid_times = [t for t in current_size_times if t is not None and t > 0]
        average_time = sum(valid_times) / len(valid_times) if valid_times else 0.0

        print(f"--- Results for size (nrows) {size} ---")
        print(f"  Individual run times (seconds): {current_size_times}")
        print(f"  Average time over {len(valid_times)} successful run(s): {average_time:.4f} seconds")

        all_results_data.append({
            "size_nrows": size,
            "times": current_size_times,
            "average": average_time,
            "successful_runs": len(valid_times)
        })
        print("-" * 60)

    # --- Save results to file ---
    print(f"\nSaving detailed results to {RESULTS_FILE}...")
    try:
        with open(RESULTS_FILE, 'w') as f:
            f.write(f"Movie Performance Test Results (using nrows loading, user-specified run method)\n") # Updated title
            f.write(f"Date: {datetime.datetime.now():%Y-%m-%d %H:%M:%S}\n")
            f.write(f"Number of runs per size (n): {NUM_RUNS}")
            f.write(f"Data sizes tested (nrows used for CSV loading): {DATA_SIZES}\n")
            f.write(f"Test Question(s): {TEST_SET_QUESTIONS}\n") # Log questions tested
            f.write(f"Ratings Data Source: {RATINGS_CSV_PATH}\n")
            f.write(f"Movies Data Source: {MOVIES_CSV_PATH}\n")
            f.write(f"LLM Model: {llm.model}\n")
            f.write(f"Neo4j Database: {NEO4J_DB}\n")
            f.write("="*60 + "\n\n")

            for result in all_results_data:
                f.write(f"Data Size (nrows): {result['size_nrows']}\n")
                f.write(f"  Individual Run Times (seconds): {result['times']}\n")
                f.write(f"  Successful Runs: {result['successful_runs']}/{NUM_RUNS}\n")
                f.write(f"  Average Time (successful runs): {result['average']:.4f} seconds\n")
                f.write("-" * 50 + "\n")

            print(f"Results saved successfully to {RESULTS_FILE}")
    except IOError as e:
        print(f"Error writing results to file: {e}", file=sys.stderr)
    except Exception as e:
         print(f"An unexpected error occurred during results saving: {e}", file=sys.stderr)


    print("\nPerformance test suite finished.")