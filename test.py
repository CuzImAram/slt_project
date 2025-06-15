import os
import pandas as pd
from dotenv import dotenv_values
from elasticsearch import Elasticsearch

# --- Pandas Anzeigeoptionen anpassen ---
# Stellt sicher, dass der gesamte Text in den Spalten angezeigt wird.
pd.set_option('display.max_colwidth', None)
# Stellt sicher, dass alle Spalten angezeigt werden.
pd.set_option('display.max_columns', None)
# Stellt sicher, dass die Ausgabe die Konsolenbreite optimal nutzt.
pd.set_option('display.width', None)


# --- Konfiguration ---
# Lädt Umgebungsvariablen aus der .env-Datei.
# Stellen Sie sicher, dass Ihre .env-Datei den ES_API_KEY enthält.
config = dotenv_values(".env")
ES_API_KEY = config.get("ES_API_KEY")

# Verbindungsparameter aus Ihrem bereitgestellten Notebook.
ES_HOST = "https://elasticsearch.bw.webis.de:9200"
INDEX_NAME_SERPS = "aql_serps"
INDEX_NAME_RESULTS = "aql_results"


class SourceRetriever:
    """
    Eine Klasse zum Abrufen von Quellen aus Elasticsearch unter Verwendung erweiterter Abfragen
    und Pandas zur Datenmanipulation. Diese Version verwendet einen Instanz-Client für ES.
    """

    def __init__(self, host: str, api_key: str, serps_index: str, results_index: str):
        """
        Initialisiert den Retriever und stellt die Verbindung zu Elasticsearch her.
        """
        self.serps_index = serps_index
        self.results_index = results_index
        self.es_client = None  # Client als None initialisieren

        if not api_key:
            print("❌ ES_API_KEY nicht in .env-Datei gefunden.")
            return

        try:
            print("Verbinde mit Elasticsearch...")
            # Speichert den Client als Instanzvariable
            self.es_client = Elasticsearch(host, api_key=api_key, verify_certs=True, request_timeout=30)
            # Verbindung testen
            self.es_client.ping()
            print("✅ Erfolgreich mit Elasticsearch verbunden!")
        except Exception as e:
            print(f"❌ Verbindung zu Elasticsearch fehlgeschlagen: {e}")
            print(
                "Bitte stellen Sie sicher, dass Sie mit dem Webis VPN verbunden sind und Ihr API-Schlüssel korrekt ist.")
            self.es_client = None

    def get_serps(self, rag_query: str):
        """
        Ruft relevante SERPs basierend auf der Benutzeranfrage ab.
        Dies ist nun eine ordnungsgemäße Instanzmethode.
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
            "size": 10_000
        }
        # Verwendet den Instanz-Client 'self.es_client'
        serps = self.es_client.search(index=self.serps_index, body=query)
        serps_df = pd.json_normalize(serps['hits']['hits']).loc[:, ["_id", "_source.warc_query", "_score"]]
        return serps_df

    def get_texts_from_index(self, serps_df: pd.DataFrame):
        """
        Ruft die Snippet-Texte für das angegebene SERPs-DataFrame ab.
        Dies ist nun eine ordnungsgemäße Instanzmethode.
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
        # Verwendet den Instanz-Client 'self.es_client'
        texts = self.es_client.search(index=self.results_index, body=query)
        texts_df = pd.json_normalize(texts['hits']['hits']).loc[:,
                           ["_source.serp.id", "_source.snippet.id", "_source.snippet.text", "_source.snippet.rank"]]
        return texts_df

    def get_context(self, rag_query: str):
        """
        Öffentliche Hauptmethode zum Abrufen, Zusammenführen und Verarbeiten von Daten
        in einem finalen Kontext-DataFrame.
        """
        if not self.es_client:
            print("Kontext kann nicht abgerufen werden, Elasticsearch-Client ist nicht verbunden.")
            return pd.DataFrame()

        # Ruft die anderen Methoden mit 'self' auf
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


# --- Anwendungsbeispiel ---
if __name__ == '__main__':
    # 1. Initialisieren Sie die Retriever-Klasse
    retriever = SourceRetriever(
        host=ES_HOST,
        api_key=ES_API_KEY,
        serps_index=INDEX_NAME_SERPS,
        results_index=INDEX_NAME_RESULTS
    )

    # 2. Fahren Sie nur fort, wenn die Verbindung erfolgreich war
    if retriever.es_client:
        user_question = "pizza pineapple"
        print(f"\n--- Rufe Kontext für die Anfrage ab: '{user_question}' ---")

        # 3. Rufen Sie die Hauptmethode auf, um den Kontext zu erhalten
        context_df = retriever.get_context(user_question)

        # 4. Zeigen Sie die Ergebnisse an
        if not context_df.empty:
            print("\n--- Abgerufener Kontext-DataFrame ---")
            print(context_df.head(100)) # Changed to head(100) to show more rows by default
            print(f"\nGesamtzahl der abgerufenen Snippets: {len(context_df)}")
            print("-----------------------------------")
        else:
            print("\n--- Es wurde kein Kontext abgerufen. ---")
