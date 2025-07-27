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

    def get_domain_counts(self, rag_query: str, size: int = 20):
        """
        Retrieves domain counts aggregation for a given query to identify top providers.

        Args:
            rag_query (str): The user query string.
            size (int): Number of top domains to return (default: 20).

        Returns:
            pd.DataFrame: DataFrame containing domain counts.
        """
        if not self.es_client:
            print("Domain counts cannot be retrieved, Elasticsearch client is not connected.")
            return pd.DataFrame()

        query = {
            "size": 0,  # We don't need the actual results, only aggregations
            "query": {
                "multi_match": {
                    "query": rag_query,
                    "fields": ["warc_query", "url_query"]
                }
            },
            "aggs": {
                "domain_counts": {
                    "terms": {
                        "field": "provider.domain",  # Field containing domain names
                        "size": size,                # Number of top domains to return
                        "order": {
                            "_count": "desc"         # Sort by frequency (descending)
                        }
                    }
                }
            }
        }

        try:
            response = self.es_client.search(index=self.serps_index, body=query)
            domain_data = response["aggregations"]["domain_counts"]["buckets"]

            domain_df = pd.DataFrame(domain_data).rename({
                "key": "domain",
                "doc_count": "count"
            }, axis=1)

            return domain_df
        except Exception as e:
            print(f"Error retrieving domain counts: {e}")
            return pd.DataFrame()

    def get_serps_with_top_providers(self, rag_query: str, top_n_providers: int = 20):
        """
        Retrieves relevant SERPs based on the user query, prioritizing top providers.

        Args:
            rag_query (str): The user query string.
            top_n_providers (int): Number of top providers to prioritize (default: 5).

        Returns:
            pd.DataFrame: DataFrame containing SERPs with provider priority.
        """
        if not self.es_client:
            print("SERPs cannot be retrieved, Elasticsearch client is not connected.")
            return pd.DataFrame()

        # First, get top domains for this query
        domain_counts = self.get_domain_counts(rag_query, size=top_n_providers)

        if domain_counts.empty:
            # Fallback to original method if no domain data
            return self.get_serps(rag_query)

        top_domains = domain_counts["domain"].tolist()

        # Enhanced query that boosts results from top providers
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
                                        "field": "warc_snippets.id"
                                    }
                                }
                            }
                        }
                    ],
                    "should": [
                        {
                            "terms": {
                                "provider.domain": top_domains,
                                "boost": 2.0  # Boost results from top providers
                            }
                        }
                    ]
                }
            },
            "size": 10
        }

        try:
            serps = self.es_client.search(index=self.serps_index, body=query)
            serps_df = pd.json_normalize(serps['hits']['hits']).loc[:,
                       ["_id", "_source.warc_query", "_source.provider.domain", "_score"]]
            return serps_df
        except Exception as e:
            print(f"Error retrieving SERPs with top providers: {e}")
            return self.get_serps(rag_query)  # Fallback to original method

    def get_serps(self, rag_query: str, use_provider_priority: bool = True, top_n_providers: int = 20):
        """
        Retrieves relevant SERPs based on the user query, with optional provider prioritization.

        Args:
            rag_query (str): The user query string.
            use_provider_priority (bool): Whether to prioritize top providers (default: True).
            top_n_providers (int): Number of top providers to prioritize (default: 5).

        Returns:
            pd.DataFrame: DataFrame containing SERPs.
        """
        if not self.es_client:
            print("SERPs cannot be retrieved, Elasticsearch client is not connected.")
            return pd.DataFrame()

        # Base query structure
        base_query = {
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

        # If provider priority is enabled, enhance the query
        if use_provider_priority:
            try:
                # First, get domain counts using aggregation
                domain_agg_query = {
                    "size": 0,  # We don't need the actual results, only aggregations
                    "query": {
                        "multi_match": {
                            "query": rag_query,
                            "fields": ["warc_query", "url_query"]
                        }
                    },
                    "aggs": {
                        "domain_counts": {
                            "terms": {
                                "field": "provider.domain",  # Field containing domain names
                                "size": top_n_providers,     # Number of top domains to return
                                "order": {
                                    "_count": "desc"          # Sort by frequency (descending)
                                }
                            }
                        }
                    }
                }

                # Execute domain aggregation query
                domain_response = self.es_client.search(index=self.serps_index, body=domain_agg_query)
                domain_data = domain_response["aggregations"]["domain_counts"]["buckets"]

                if domain_data:  # If we have domain data, enhance the query
                    top_domains = [bucket["key"] for bucket in domain_data]

                    # Add provider boosting to the base query
                    base_query["query"]["bool"]["should"] = [
                        {
                            "terms": {
                                "provider.domain": top_domains,
                                "boost": 2.0  # Boost results from top providers
                            }
                        }
                    ]

            except Exception as e:
                print(f"Warning: Could not retrieve domain counts, falling back to standard query: {e}")
                # Continue with base query without provider priority

        try:
            serps = self.es_client.search(index=self.serps_index, body=base_query)

            # Include provider domain in the result if available
            columns = ["_id", "_source.warc_query", "_score"]
            try:
                # Try to include provider domain if it exists in the results
                serps_df = pd.json_normalize(serps['hits']['hits'])
                if "_source.provider.domain" in serps_df.columns:
                    columns.append("_source.provider.domain")
                serps_df = serps_df.loc[:, columns]
            except (KeyError, IndexError):
                # Fallback to basic columns if provider domain is not available
                serps_df = pd.json_normalize(serps['hits']['hits']).loc[:, ["_id", "_source.warc_query", "_score"]]

            return serps_df

        except Exception as e:
            print(f"Error retrieving SERPs: {e}")
            return pd.DataFrame()

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

    def get_context_with_provider_priority(self, rag_query: str, top_n_providers: int = 20):
        """
        Enhanced context retrieval method that prioritizes top providers.

        Args:
            rag_query (str): The user query string.
            top_n_providers (int): Number of top providers to prioritize.

        Returns:
            pd.DataFrame: DataFrame containing the final context with provider information.
        """
        if not self.es_client:
            print("Context cannot be retrieved, Elasticsearch client is not connected.")
            return pd.DataFrame()

        # Get SERPs with provider priority
        serps = self.get_serps_with_top_providers(rag_query, top_n_providers)
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
                "_source.provider.domain": "provider_domain",
                "_source.serp.id": "serp_id",
                "_source.snippet.id": "snippet_id",
                "_source.snippet.text": "text",
                "_source.snippet.rank": "rank",
            }, axis=1)
            .sort_values(["score", "rank"], ascending=[False, True])
            .assign(length=lambda df: df["text"].apply(len)).query("length > 100")
            .loc[:, ["query", "provider_domain", "text"]]
            .reset_index(drop=True)
        )
        return context

    def get_context(self, rag_query: str, use_provider_priority: bool = True):
        """
        Public main method to retrieve, merge, and process data
        into a final context DataFrame.

        Args:
            rag_query (str): The user query string.
            use_provider_priority (bool): Whether to use provider prioritization (default: True).

        Returns:
            pd.DataFrame: DataFrame containing the final context.
        """
        if not self.es_client:
            print("Context cannot be retrieved, Elasticsearch client is not connected.")
            return pd.DataFrame()

        # Call get_serps with provider priority option
        serps = self.get_serps(rag_query, use_provider_priority=use_provider_priority)
        texts = self.get_texts_from_index(serps)

        # Check if provider domain is available in serps
        has_provider_domain = "_source.provider.domain" in serps.columns

        context = (
            pd.merge(
                serps,
                texts,
                left_on="_id",
                right_on="_source.serp.id",
                how="inner"
            )
            .drop("_id", axis=1)
        )

        # Define rename dictionary based on available columns
        rename_dict = {
            "_source.warc_query": "query",
            "_score": "score",
            "_source.serp.id": "serp_id",
            "_source.snippet.id": "snippet_id",
            "_source.snippet.text": "text",
            "_source.snippet.rank": "rank",
        }

        if has_provider_domain:
            rename_dict["_source.provider.domain"] = "provider_domain"

        context = context.rename(rename_dict, axis=1)

        # Define final columns based on what's available
        final_columns = ["query", "text"]
        if has_provider_domain and "provider_domain" in context.columns:
            final_columns = ["query", "provider_domain", "text"]

        context = (
            context
            .sort_values(["score", "rank"], ascending=[False, True])
            .assign(length=lambda df: df["text"].apply(len))
            .query("length > 100")  # This can also be done in Elastic directly – try to figure out how!
            .loc[:, final_columns]
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

        # 3a. First, show domain counts for the query
        print("\n--- Top Provider Domains for this query ---")
        domain_counts = retriever.get_domain_counts(user_question)
        if not domain_counts.empty:
            print(domain_counts.head(10))
        else:
            print("No domain data found.")

        # 3b. Call the enhanced method with provider priority
        print(f"\n--- Using provider-prioritized context retrieval ---")
        context_df_enhanced = retriever.get_context_with_provider_priority(user_question, top_n_providers=5)

        # 3c. Call the original method for comparison
        print(f"\n--- Using original context retrieval ---")
        context_df_original = retriever.get_context(user_question)

        # 4. Display and compare the results
        if not context_df_enhanced.empty:
            print("\n--- Enhanced context DataFrame (with provider priority) ---")
            print(context_df_enhanced.head(10))
            print(f"\nTotal number of retrieved snippets (enhanced): {len(context_df_enhanced)}")

            if 'provider_domain' in context_df_enhanced.columns:
                print("\n--- Provider domain distribution in enhanced results ---")
                print(context_df_enhanced['provider_domain'].value_counts().head(10))
            print("-----------------------------------")
        else:
            print("\n--- No enhanced context was retrieved. ---")

        if not context_df_original.empty:
            print("\n--- Original context DataFrame ---")
            print(context_df_original.head(10))
            print(f"\nTotal number of retrieved snippets (original): {len(context_df_original)}")
            print("-----------------------------------")
        else:
            print("\n--- No original context was retrieved. ---")
