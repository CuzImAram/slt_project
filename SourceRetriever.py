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
            "size": 50
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
                                "boost": 2.0
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
        if serps_df.empty:
            print("Warning: No SERPs provided to get_texts_from_index")
            return pd.DataFrame()

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

        try:
            texts = self.es_client.search(index=self.results_index, body=query)

            if not texts['hits']['hits']:
                print("Warning: No texts found for the provided SERP IDs")
                return pd.DataFrame()

            # Normalize the JSON response
            texts_df = pd.json_normalize(texts['hits']['hits'])

            # Check if the required columns exist
            required_columns = ["_source.serp.id", "_source.snippet.id", "_source.snippet.text", "_source.snippet.rank"]
            available_columns = texts_df.columns.tolist()

            missing_columns = [col for col in required_columns if col not in available_columns]
            if missing_columns:
                print(f"Warning: Missing columns in texts response: {missing_columns}")
                print(f"Available columns: {available_columns}")
                return pd.DataFrame()

            # Select only the required columns
            texts_df = texts_df.loc[:, required_columns]
            return texts_df

        except Exception as e:
            print(f"Error in get_texts_from_index: {e}")
            return pd.DataFrame()

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

    def get_context_pipeline4(self, query_list: list, use_provider_priority: bool = True):
        """
        Pipeline method that takes a list of space-separated queries and uses
        Elasticsearch default_operator AND for better precision.

        Args:
            query_list (list): List of space-separated query strings from LLM.
            use_provider_priority (bool): Whether to use provider prioritization (default: True).

        Returns:
            pd.DataFrame: Combined context DataFrame from all queries.
        """
        if not self.es_client:
            print("Context cannot be retrieved, Elasticsearch client is not connected.")
            return pd.DataFrame()

        if not query_list:
            print("No queries provided to pipeline.")
            return pd.DataFrame()

        print(f"\n--- Starting Pipeline 4 with {len(query_list)} queries ---")

        all_contexts = []

        for i, space_query in enumerate(query_list):
            print(f"Query {i+1}: '{space_query}' -> Elasticsearch with default_operator AND")

            # Get context for this space-separated query using default_operator AND
            context = self.get_serps_with_and_query(space_query, use_provider_priority)
            
            if context.empty:
                print(f"  ⚠️ No results found with AND, retrying with OR")
                context = self.get_context(space_query, use_provider_priority)


            if not context.empty:
                # Add query information for tracking
                context['source_query'] = space_query
                all_contexts.append(context)
                print(f"  ✅ Found {len(context)} results")
            else:
                print(f"  ⚠️ No results found")

        if all_contexts:
            # Combine all contexts and remove duplicates based on text
            combined_context = pd.concat(all_contexts, ignore_index=True)

            # Remove duplicate texts while keeping track of source queries
            combined_context = combined_context.drop_duplicates(subset=['text'], keep='first')

            # Sort by score if available
            if 'score' in combined_context.columns:
                combined_context = combined_context.sort_values('score', ascending=False)

            combined_context = combined_context.reset_index(drop=True)
            print(f"✅ Pipeline 4 completed: {len(combined_context)} unique results from {len(query_list)} queries")

            return combined_context
        else:
            print("⚠️ No results found from any query in pipeline")
            return pd.DataFrame()

    def get_serps_with_and_query(self, query: str, use_provider_priority: bool = True):
        """
        Helper method to get context using Elasticsearch query_string with default_operator AND.

        Args:
            query (str): Space-separated query string (e.g., "Sam Altman AI").
            use_provider_priority (bool): Whether to use provider prioritization.

        Returns:
            pd.DataFrame: DataFrame containing context results.
        """
        if not self.es_client:
            return pd.DataFrame()

        # Base query structure using default_operator AND
        base_query = {
            "query": {
                "bool": {
                    "must": [
                        {
                            "query_string": {
                                "query": query,
                                "fields": ["warc_query"],
                                "default_operator": "AND"
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
                    ]
                }
            },
            "size": 50
        }

        # Add provider priority if enabled
        if use_provider_priority:
            try:
                # Get domain counts using aggregation
                domain_agg_query = {
                    "size": 0,
                    "query": {
                        "query_string": {
                            "query": query,
                            "fields": ["warc_query", "url_query"],
                            "default_operator": "AND"
                        }
                    },
                    "aggs": {
                        "domain_counts": {
                            "terms": {
                                "field": "provider.domain",
                                "size": 20,
                                "order": {
                                    "_count": "desc"
                                }
                            }
                        }
                    }
                }

                domain_response = self.es_client.search(index=self.serps_index, body=domain_agg_query)
                domain_data = domain_response["aggregations"]["domain_counts"]["buckets"]

                if domain_data:
                    top_domains = [bucket["key"] for bucket in domain_data]
                    base_query["query"]["bool"]["should"] = [
                        {
                            "terms": {
                                "provider.domain": top_domains,
                                "boost": 2.0
                            }
                        }
                    ]

            except Exception as e:
                print(f"Warning: Could not retrieve domain counts for AND query, using standard query: {e}")

        try:
            # Get SERPs using query_string with default_operator AND
            serps_response = self.es_client.search(index=self.serps_index, body=base_query)

            if not serps_response['hits']['hits']:
                return pd.DataFrame()

            # Process SERP results
            columns = ["_id", "_source.warc_query", "_score"]
            try:
                serps_df = pd.json_normalize(serps_response['hits']['hits'])
                if "_source.provider.domain" in serps_df.columns:
                    columns.append("_source.provider.domain")
                serps_df = serps_df.loc[:, columns]
            except (KeyError, IndexError):
                serps_df = pd.json_normalize(serps_response['hits']['hits']).loc[:, ["_id", "_source.warc_query", "_score"]]

            # Get texts for these SERPs
            texts_df = self.get_texts_from_index(serps_df)

            if texts_df.empty:
                return pd.DataFrame()

            # Merge and process
            has_provider_domain = "_source.provider.domain" in serps_df.columns

            context = pd.merge(
                serps_df,
                texts_df,
                left_on="_id",
                right_on="_source.serp.id",
                how="inner"
            ).drop("_id", axis=1)

            # Rename columns
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

            # Final processing
            final_columns = ["query", "text"]
            if has_provider_domain and "provider_domain" in context.columns:
                final_columns = ["query", "provider_domain", "text"]

            context = (
                context
                .sort_values(["score", "rank"], ascending=[False, True])
                .assign(length=lambda df: df["text"].apply(len))
                .query("length > 100")
                .loc[:, final_columns]
                .reset_index(drop=True)
            )

            return context

        except Exception as e:
            print(f"Error in AND query search: {e}")
            return pd.DataFrame()