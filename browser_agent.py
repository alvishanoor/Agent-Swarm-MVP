# browser_agent.py

import os
import requests
# Using SerpApi instead of googlesearch to bypass IP blocking
from serpapi import GoogleSearch 
import time
from typing import List, Dict, Any
from bs4 import BeautifulSoup # Kept for get_page_title functionality (if needed)

class BrowserAgent:
    """
    Specialized agent for web search and research using SerpApi to avoid IP blocking issues.
    The PAEI specific search modification is handled in the main function when calling search_web.
    """
    def __init__(self):
        self.headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        
        # Load API Key from environment variable (set in main.py)
        self.api_key = os.getenv("SERPAPI_KEY") 
        
        if not self.api_key:
             print("❌ ERROR: SerpApi Key not loaded from environment.") 
        else:
             print("✅ SerpApi Key loaded successfully.")
        
    def search_web(self, query: str, num_results: int = 5) -> List[Dict[str, Any]]:
        """Web search using SerpApi."""
        
        if not self.api_key:
            return []

        print(f"🔍 Searching via SerpApi for: {query}")

        params = {
            "api_key": self.api_key,
            "engine": "google",
            "q": query,
            "hl": "en",
            "gl": "us",
            "num": num_results # SerpApi handles the number of results limit
        }

        try:
            search = GoogleSearch(params)
            results_data = search.get_dict()
            
            results = []
            
            # Extract organic results
            if "organic_results" in results_data:
                for i, res in enumerate(results_data["organic_results"][:num_results]):
                    results.append({
                        'rank': i + 1,
                        # SerpApi provides title and snippet directly, improving quality
                        'url': res.get('link', '#'),
                        'title': res.get('title', 'No Title'),
                        'snippet': res.get('snippet', 'No description available.')
                    })
            
            return results

        except Exception as e:
            print(f"SerpApi Search Error: {e}")
            return []

    def get_page_title(self, url):
        """Get title of a webpage (Kept for robustness, but SerpApi already provides it)."""
        # This function is now largely redundant but kept in the class structure.
        return url
    
    def format_search_results(self, results: List[Dict[str, Any]]) -> str:
        """
        Formats the search results into a clean, readable string for the LLM.
        This is crucial for the Parent Agent (Gemini) to process the data effectively.
        """
        if not results:
            return "Search failed: Could not retrieve any relevant information from the web."

        formatted_output = "--- START OF SEARCH RESULTS ---\n"
        for i, res in enumerate(results):
            # Limit snippet length for clarity and token efficiency
            snippet = res.get('snippet', 'No description available.')
            if len(snippet) > 150:
                snippet = snippet[:147] + "..."
                
            formatted_output += (
                f"RESULT {i+1} | Title: {res.get('title', 'N/A')}\n"
                f"Snippet: {snippet}\n"
                f"Source URL: {res.get('url', 'N/A')}\n"
                f"---------------------------------\n"
            )
        formatted_output += "--- END OF SEARCH RESULTS ---"
        return formatted_output

    def research_for_agent(self, query: str, agent_type: str) -> List[Dict[str, Any]]:
        """Customize search based on PAEI agent type and calls the search function."""
        if agent_type == "Producer":
            query += " quick guide how to complete"
        elif agent_type == "Administrator":
            query += " step by step process tutorial"
        elif agent_type == "Entrepreneur":
            query += " strategy trends innovation future"
        elif agent_type == "Integrator":
            query += " team collaboration relationships"
        
        # Now uses the new SerpApi based search
        return self.search_web(query)