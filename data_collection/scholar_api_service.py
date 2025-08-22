# # fetch_scholar_data.py

# import os
# import time
# import json
# import requests
# import argparse
# from dotenv import load_dotenv
# from typing import List, Dict, Optional, Any

# # Load environment variables (specifically the API key)
# load_dotenv()

# S2_API_BASE_URL = "https://api.semanticscholar.org/graph/v1"
# S2_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY")

# class SemanticScholarAPI:
#     """
#     A robust client to interact with the Semantic Scholar (S2) Graph API.
#     Handles the "search-then-disambiguate" workflow correctly.
#     """
#     def __init__(self, api_key: str):
#         if not api_key:
#             raise ValueError("API Key is required to use this service.")
#         self.api_key = api_key
#         self.headers = {'x-api-key': self.api_key}
#         print("Semantic Scholar API client initialized with API key.")

#     def _make_request(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
#         """Internal helper to make requests, handle errors, and respect rate limits."""
#         time.sleep(1) # Respect the 1 request/second rate limit
#         url = f"{S2_API_BASE_URL}/{endpoint}"
#         try:
#             response = requests.get(url, params=params, headers=self.headers)
#             response.raise_for_status()
#             return response.json()
#         except requests.exceptions.HTTPError as http_err:
#             print(f"HTTP error occurred: {http_err}")
#             print(f"Server Response Content: {http_err.response.content.decode()}")
#         except requests.exceptions.RequestException as req_err:
#             print(f"Request error occurred: {req_err}")
#         return None

#     def search_authors(self, name_query: str) -> List[Dict[str, Any]]:
#         """
#         Searches for authors by NAME ONLY.
#         """
#         # --- THE FIX IS HERE ---
#         # The 'query' parameter should only contain the name.
#         # Punctuation is removed to be safe.
#         sanitized_name = name_query.replace('.', '')
#         print(f"Searching for author name: '{sanitized_name}'...")
        
#         params = {
#             'query': sanitized_name,
#             'fields': 'name,paperCount,hIndex,affiliations', # affiliations is a supported field to get back
#             'limit': 5 # Get top 5 potential matches
#         }
        
#         data = self._make_request('author/search', params)
#         return data.get('data', []) if data else []

#     def get_author_details(self, author_id: str) -> Optional[Dict[str, Any]]:
#         """Fetches detailed information for a specific author using their unique S2 ID."""
#         print(f"Fetching details for author ID: {author_id}...")
        
#         fields = (
#             'name,affiliations,paperCount,citationCount,hIndex,'
#             'papers.title,papers.year,papers.abstract,papers.venue,'
#             'papers.citationCount,papers.authors'
#         )
#         params = {'fields': fields}
#         endpoint = f"author/{author_id}"
        
#         author_data = self._make_request(endpoint, params)
#         if author_data:
#             print(f"Successfully fetched details for {author_data.get('name')}.")
#             return self._structure_author_data(author_data)
#         return None

#     def _structure_author_data(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
#         """Parses the raw API response into a cleaner, structured dictionary."""
#         # ... (This function is correct and remains the same) ...
#         author_id = raw_data.get('authorId')
#         structured_info = {
#             's2_author_id': author_id, 'name': raw_data.get('name'),
#             'affiliations': raw_data.get('affiliations', []), 'paper_count': raw_data.get('paperCount'),
#             'citation_count': raw_data.get('citationCount'), 'h_index': raw_data.get('hIndex'),
#             'papers': [], 'co_authors': {}
#         }
#         if 'papers' in raw_data:
#             for paper in raw_data.get('papers', []):
#                 structured_info['papers'].append({
#                     's2_paper_id': paper.get('paperId'), 'title': paper.get('title'),
#                     'year': paper.get('year'), 'venue': paper.get('venue'),
#                     'citation_count': paper.get('citationCount'), 'abstract': paper.get('abstract')
#                 })
#                 if 'authors' in paper:
#                     for co_author in paper.get('authors', []):
#                         co_author_id = co_author.get('authorId')
#                         if co_author_id and co_author_id != author_id:
#                             structured_info['co_authors'][co_author_id] = co_author.get('name')
#         structured_info['co_authors'] = [
#             {'s2_author_id': id, 'name': name} for id, name in structured_info['co_authors'].items()
#         ]
#         return structured_info

# # --- Main function to run the script from the command line ---
# if __name__ == "__main__":
#     if not S2_API_KEY:
#         print("FATAL ERROR: SEMANTIC_SCHOLAR_API_KEY not found in your .env file.")
#         exit()

#     parser = argparse.ArgumentParser(description="Fetch detailed researcher data from Semantic Scholar.")
#     parser.add_argument("name", help="Full name of the researcher to search for.")
#     parser.add_argument("-a", "--affiliation", help="Optional: Affiliation to help you choose the correct person.")
#     args = parser.parse_args()

#     s2_api = SemanticScholarAPI(api_key=S2_API_KEY)

#     # 1. Search for author by NAME ONLY
#     candidates = s2_api.search_authors(args.name)

#     if not candidates:
#         print(f"\nNo potential matches found for '{args.name}'. Exiting.")
#         exit()

#     # 2. Present the choices to the user, using the affiliation for guidance
#     print("\n--- Found Potential Matches ---")
#     for i, candidate in enumerate(candidates):
#         affiliations = ", ".join(candidate.get('affiliations', [])) or "N/A"
        
#         # --- NEW DISAMBIGUATION LOGIC ---
#         # Highlight candidates that match the provided affiliation
#         match_indicator = ""
#         if args.affiliation and affiliations != "N/A":
#             if any(args.affiliation.lower() in aff.lower() for aff in candidate.get('affiliations', [])):
#                 match_indicator = "  <-- AFFILIATION MATCH!"
        
#         print(f"  [{i+1}] {candidate.get('name')} | Papers: {candidate.get('paperCount', 0)} | Affiliations: {affiliations}{match_indicator}")
    
#     try:
#         choice_str = input("\nPlease choose the correct author by number (or 0 to exit): ")
#         choice = int(choice_str)
#         if choice == 0 or choice > len(candidates):
#             print("Exiting.")
#             exit()
        
#         chosen_author = candidates[choice - 1]
#         author_id = chosen_author['authorId']
#         print(f"You chose: {chosen_author.get('name')} (ID: {author_id})")

#     except (ValueError, IndexError):
#         print("Invalid choice. Exiting.")
#         exit()

#     # 3. Fetch Detailed Data for the Chosen Author
#     author_details = s2_api.get_author_details(author_id)

#     # 4. Save the Data to a JSON file
#     if author_details:
#         filename = f"{author_details['name'].replace(' ', '_').lower()}_s2_data.json"
#         with open(filename, 'w') as f:
#             json.dump(author_details, f, indent=4)
#         print(f"\n--- SUCCESS ---")
#         print(f"Full researcher data has been saved to: {filename}")

# scholar_api_service.py

import os
import time
import requests
from dotenv import load_dotenv
from typing import List, Dict, Optional, Any

load_dotenv()
S2_API_BASE_URL = "https://api.semanticscholar.org/graph/v1"
S2_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY")

class SemanticScholarAPI:
    """
    A robust and efficient client for the Semantic Scholar API.
    Designed to avoid rate-limiting errors during batch processing.
    """
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("API Key is required for this service.")
        self.api_key = api_key
        self.headers = {'x-api-key': self.api_key}
        print("Semantic Scholar API client initialized with API key.")

    def _make_request(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """Internal helper to make requests, handle errors, and respect rate limits."""
        time.sleep(1.1) # Wait 1.1 seconds to be safely within the 1 req/sec limit
        url = f"{S2_API_BASE_URL}/{endpoint}"
        try:
            response = requests.get(url, params=params, headers=self.headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as http_err:
            print(f"HTTP error occurred: {http_err}")
            print(f"Server Response Content: {http_err.response.content.decode()}")
        except requests.exceptions.RequestException as req_err:
            print(f"Request error occurred: {req_err}")
        return None

    def search_authors_with_content(self, name_query: str) -> List[Dict[str, Any]]:
        """
        Searches for authors and efficiently fetches their paper titles/abstracts
        in the SAME API call to prevent rate-limiting.
        """
        sanitized_name = name_query.replace('.', '')
        
        # --- KEY CHANGE FOR EFFICIENCY ---
        # We now ask for paper titles and abstracts directly in the search.
        # This reduces API calls from (1+N) to just 1 per researcher.
        params = {
            'query': sanitized_name,
            'fields': 'name,paperCount,hIndex,affiliations,papers.title,papers.abstract',
            'limit': 5
        }
        
        data = self._make_request('author/search', params)
        return data.get('data', []) if data else []

    def get_author_details(self, author_id: str) -> Optional[Dict[str, Any]]:
        """Fetches the full profile for a chosen author."""
        fields = (
            'name,affiliations,paperCount,citationCount,hIndex,'
            'papers.title,papers.year,papers.venue,papers.citationCount,papers.authors,papers.abstract'
        )
        params = {'fields': fields}
        endpoint = f"author/{author_id}"
        return self._make_request(endpoint, params)
    
    @staticmethod
    def find_best_author_match(candidates: list, nsf_profile: dict) -> Optional[dict]:
        """
        Uses scoring with topical overlap to find the best match.
        This version DOES NOT make any new API calls.
        """
        if not candidates: return None

        best_match = None
        highest_score = -1
        
        nsf_keywords = set(nsf_profile.get('keywords', []))
        nsf_affiliation = nsf_profile.get('primary_affiliation', '')
        
        print(f"  -> Matching against NSF Keywords: {nsf_keywords}")

        for candidate in candidates:
            score = 0
            
            # --- KEY CHANGE FOR ROBUSTNESS & EFFICIENCY ---
            # Create the content fingerprint from the data we *already* fetched.
            s2_papers = candidate.get('papers', [])
            s2_content_parts = []
            for paper in s2_papers:
                # Safely handle None values for title or abstract
                title = paper.get('title') or ""
                abstract = paper.get('abstract') or ""
                s2_content_parts.append(title)
                s2_content_parts.append(abstract)
            s2_content = " ".join(s2_content_parts).lower()
            
            if s2_content:
                topical_overlap = sum(1 for keyword in nsf_keywords if keyword in s2_content)
                score += topical_overlap * 10
                print(f"    - Candidate '{candidate['name']}' | Topical Score = {topical_overlap * 10}")

            if nsf_affiliation and candidate.get('affiliations'):
                if any(nsf_affiliation.lower() in aff.lower() for aff in candidate['affiliations']):
                    score += 100
                    print(f"    - Candidate '{candidate['name']}' | Affiliation match! (+100 points)")

            score += candidate.get('paperCount', 0) * 0.1

            if score > highest_score:
                highest_score = score
                best_match = candidate
        
        if best_match:
            print(f"  -> Best Match Selected: '{best_match['name']}' with a score of {highest_score:.2f}")

        return best_match
    
"""
REQUIREMENTS:
1. use all the NSF data
2. this will be used for hiring teacher like qunatam computing, Machine learning and so on
3. the possible questions I could ask is, who are the top profesors/researchers in * department/topic
4. from the results, I could check one researcher and get all relevent info from there
5. related info can consist colaborators, reserach, papers and other related topics
6. to get the related info, check if any other data source is available or not (like getting data from google scholer)
7. if possile interface neo4j in web portal or some way to visulalize intaractive graphs of the reseracher info
"""