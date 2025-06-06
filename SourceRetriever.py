import pandas as pd
from elasticsearch import Elasticsearch
from dotenv import dotenv_values

# Load environment variables from the .env file
# Make sure your .env file contains the ES_API_KEY.
config = dotenv_values(".env")

# --- Configuration ---
# Connection parameters, as in notebook 01-elasticsearch-solved.ipynb
ES_HOST = "https://elasticsearch.bw.webis.de:9200"
ES_API_KEY = config.get("ES_API_KEY")
INDEX_NAME_SERPS = "aql_serps"
INDEX_NAME_RESULTS = "aql_results"


class SourceRetriever:
    """
    A class to retrieve sources from Elasticsearch based on a user prompt.
    """

    def __init__(self, host: str, api_key: str, serps_index: str, results_index: str):
        """
        Initializes the retriever and establishes the connection to Elasticsearch.

        Args:
            host (str): The URL of the Elasticsearch cluster.
            api_key (str): The API key for authentication.
            serps_index (str): The name of the index containing the SERPs.
            results_index (str): The name of the index containing the result snippets.
        """
        self.serps_index = serps_index
        self.results_index = results_index
        try:
            # Connect to the client, as shown in the notebook
            self.es_client = Elasticsearch(host, api_key=api_key, verify_certs=True, request_timeout=30)
            # Test the connection
            self.es_client.search(index=INDEX_NAME_SERPS, body={"query": {"match_all": {}}})
            print("✅ Successfully connected to Elasticsearch!")
        except Exception as e:
            print(f"❌ Connection to Elasticsearch failed: {e}")
            print("Please ensure you are connected to the Webis VPN.")
            self.es_client = None

    def retrieve_sources(self, user_prompt: str, top_k_serps: int = 10) -> list[str]:
        """
        Retrieves relevant sources for a given user prompt.

        The process follows the steps from the notebook:
        1. Find relevant SERPs in the 'aql_serps' index.
        2. Use their IDs to fetch the corresponding snippets from the 'aql_results' index.
        3. Clean and filter the results.

        Args:
            user_prompt (str): The user's query.
            top_k_serps (int): The number of top SERPs to consider for source retrieval.

        Returns:
            list[str]: A list of cleaned and relevant source texts (snippets).
        """
        if not self.es_client:
            print("No Elasticsearch connection. Method cannot be executed.")
            return []

        # --- Step 1: Find relevant SERPs ---
        # Query based on the 'multi_match' and 'nested' example from the notebook
        search_query = {
            "query": {
                "bool": {
                    "must": [
                        {"multi_match": {"query": user_prompt, "fields": ["warc_query", "url_query"]}},
                        # Search in both query fields
                        {"nested": {"path": "warc_snippets", "query": {"exists": {"field": "warc_snippets.id"}}}}
                        # Ensures that snippets exist
                    ]
                }
            },
            "size": top_k_serps
        }

        try:
            serps_response = self.es_client.search(index=self.serps_index, body=search_query)
            serp_ids = [hit['_id'] for hit in serps_response['hits']['hits']]

            if not serp_ids:
                print("No SERPs with snippets found for the query.")
                return []

        except Exception as e:
            print(f"Error during SERP search: {e}")
            return []

        # --- Step 2: Retrieve corresponding snippets ---
        # Query based on 'terms' filtering by IDs from the notebook
        results_query = {
            "query": {
                "bool": {
                    "must": [
                        {"terms": {"serp.id": serp_ids}},  # Search for all relevant SERP IDs
                        {"exists": {"field": "snippet.text"}}  # Ensures that the snippet text exists
                    ]
                }
            },
            "size": 1000  # Set high enough to get all snippets
        }

        try:
            results_response = self.es_client.search(index=self.results_index, body=results_query)
        except Exception as e:
            print(f"Error during snippet search: {e}")
            return []

        # --- Step 3: Process results with Pandas ---
        # The processing is based on the last example in the notebook
        texts_df = (
            pd.json_normalize(results_response['hits']['hits'])
            .rename(columns={
                "_source.snippet.text": "text",
                "_score": "score"
            })
            .loc[:, ["text", "score"]]
            .assign(length=lambda df: df["text"].str.len())
            .query("length > 20")  # Filter very short snippets
            .sort_values("score", ascending=False)
            .drop_duplicates(subset=["text"])  # Remove exact duplicates
            .reset_index(drop=True)
        )

        return texts_df['text'].tolist()


if __name__ == '__main__':
    # Example of how to use the class

    # Check if the API key is present
    if not ES_API_KEY:
        print("Error: ES_API_KEY not found in the .env file.")
        print("Please create a '.env' file in the same directory with the content: ES_API_KEY='YourKey'")
    else:
        # 1. Initialize the retriever
        retriever = SourceRetriever(
            host=ES_HOST,
            api_key=ES_API_KEY,
            serps_index=INDEX_NAME_SERPS,
            results_index=INDEX_NAME_RESULTS
        )

        # 2. Make a request if the connection was successful
        if retriever.es_client:
            user_question = "why has olive oil increased in price"
            print(f"\nSearching for sources for the query: '{user_question}'\n")

            # 3. Retrieve sources
            sources = retriever.retrieve_sources(user_question)

            # 4. Print the results
            if sources:
                print(f"--- {len(sources)} relevant sources found ---")
                for i, source in enumerate(sources[:10]):  # Show the top 5
                    print(f"Source {i + 1}: {source}\n")
            else:
                print("No suitable sources were found.")