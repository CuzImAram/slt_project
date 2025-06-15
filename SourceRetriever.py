import pandas as pd
from dotenv import dotenv_values
from elasticsearch import Elasticsearch

# --- Adjust Pandas display options ---
# Ensures that the entire text in the columns is displayed.
pd.set_option('display.max_colwidth', None)
# Ensures that all columns are displayed.
pd.set_option('display.max_columns', None)
# Ensures that the output optimally uses the console width.
pd.set_option('display.width', None)


# --- Configuration ---
# Loads environment variables from the .env file.
# Make sure your .env file contains the ES_API_KEY.
config = dotenv_values(".env")
ES_API_KEY = config.get("ES_API_KEY")

# Connection parameters from your provided notebook.
ES_HOST = "https://elasticsearch.bw.webis.de:9200"
INDEX_NAME_SERPS = "aql_serps"
INDEX_NAME_RESULTS = "aql_results"


class SourceRetriever:
    """
    A class for retrieving sources from Elasticsearch using advanced queries
    and Pandas for data manipulation. This version uses an instance client for ES.
    """

    def __init__(self, host: str, api_key: str, serps_index: str, results_index: str):
        """
        Initializes the retriever and establishes the connection to Elasticsearch.

        Args:
            host (str): The Elasticsearch host URL.
            api_key (str): The API key for authentication.
            serps_index (str): The name of the SERPs index.
            results_index (str): The name of the results index.
        """
        self.serps_index = serps_index
        self.results_index = results_index
        self.es_client = None  # Initialize client as None

        if not api_key:
            print("❌ ES_API_KEY not found in .env file.")
            return

        try:
            print("Connecting to Elasticsearch...")
            # Store the client as an instance variable
            self.es_client = Elasticsearch(host, api_key=api_key, verify_certs=True, request_timeout=30)
            # Test connection
            self.es_client.ping()
            print("✅ Successfully connected to Elasticsearch!")
        except Exception as e:
            print(f"❌ Failed to connect to Elasticsearch: {e}")
            print(
                "Please make sure you are connected to the Webis VPN and your API key is correct.")
            self.es_client = None

    def get_serps(self, rag_query: str):
        """
        Retrieves relevant SERPs based on the user query.

        Args:
            rag_query (str): The user query string.

        Returns:
            pd.DataFrame: DataFrame containing SERPs.
        """
        query = {
            "query": {
                "bool": {
                    "must": [
                        {
                            "multi_match": {
                                "query": rag_query,
                                "fields": ["warc_query"]
                            }
                        },
                        {
                            "nested": {
                                "path": "warc_snippets",
                                "query": {
                                    "exists": {
                                        "field": "warc_snippets.id"  # Only results that have parsed results
                                    }
                                }
                            }
                        }
                    ]
                }
            },
            "size": 100
        }
        serps = self.es_client.search(index=self.serps_index, body=query)
        serps_df = pd.json_normalize(serps['hits']['hits']).loc[:, ["_id", "_source.warc_query", "_score"]]
        return serps_df

    def get_texts_from_index(self, serps_df: pd.DataFrame):
        """
        Retrieves the snippet texts for the given SERPs DataFrame.

        Args:
            serps_df (pd.DataFrame): DataFrame containing SERP IDs.

        Returns:
            pd.DataFrame: DataFrame containing snippet texts.
        """
        query = {
            "query": {
                "bool": {
                    "must": [
                        {
                            "terms": {
                                "serp.id": serps_df["_id"].values.tolist()
                                # Query multiple IDs at the same time for efficiency
                            },
                        },
                        {
                            "exists": {
                                "field": "snippet.text"  # Only results that have parsed texts
                            }
                        }
                    ]
                }
            },
            "size": 10_000  # Set to maximum, to make sure we get all results
        }
        texts = self.es_client.search(index=self.results_index, body=query)
        texts_df = pd.json_normalize(texts['hits']['hits']).loc[:,
                           ["_source.serp.id", "_source.snippet.id", "_source.snippet.text", "_source.snippet.rank"]]
        return texts_df

    def get_context(self, rag_query: str):
        """
        Public main method to retrieve, merge, and process data
        into a final context DataFrame.

        Args:
            rag_query (str): The user query string.

        Returns:
            pd.DataFrame: DataFrame containing the final context.
        """
        if not self.es_client:
            print("Context cannot be retrieved, Elasticsearch client is not connected.")
            return pd.DataFrame()

        # Calls the other methods with 'self'
        serps = self.get_serps(rag_query)
        texts = self.get_texts_from_index(serps)

        context = (
            pd.merge(
                serps,
                texts,
                left_on="_id",
                right_on="_source.serp.id",
                how="inner"
            )
            .drop("_id", axis=1)
            .rename({
                "_source.warc_query": "query",
                "_score": "score",
                "_source.serp.id": "serp_id",
                "_source.snippet.id": "snippet_id",
                "_source.snippet.text": "text",
                "_source.snippet.rank": "rank",
            }, axis=1)
            .sort_values(["score", "rank"], ascending=[False, True])
            .assign(length=lambda df: df["text"].apply(len)).query(
                "length > 100")  # This can also be done in Elastic directly – try to figure out how!
            .loc[:, ["query", "text"]]
            .reset_index(drop=True)
        )
        return context


# --- Usage example ---
if __name__ == '__main__':
    # 1. Initialize the retriever class
    retriever = SourceRetriever(
        host=ES_HOST,
        api_key=ES_API_KEY,
        serps_index=INDEX_NAME_SERPS,
        results_index=INDEX_NAME_RESULTS
    )

    # 2. Continue only if the connection was successful
    if retriever.es_client:
        user_question = "pizza pineapple"
        print(f"\n--- Retrieving context for query: '{user_question}' ---")

        # 3. Call the main method to get the context
        context_df = retriever.get_context(user_question)

        # 4. Display the results
        if not context_df.empty:
            print("\n--- Retrieved context DataFrame ---")
            print(context_df.head(100)) # Changed to head(100) to show more rows by default
            print(f"\nTotal number of retrieved snippets: {len(context_df)}")
            print("-----------------------------------")
        else:
            print("\n--- No context was retrieved. ---")

