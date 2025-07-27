import pandas as pd
from dotenv import dotenv_values
from openai import OpenAI
from SourceRetriever import SourceRetriever
import json
import re  # Import regular expressions for robust parsing

# --- Configuration ---
# Loads environment variables from the .env file.
# Make sure your .env file contains the API_KEY for the LLM.
config = dotenv_values(".env")
LLM_API_KEY = config.get("API_KEY")
LLM_API_URL = "https://api.helmholtz-blablador.fz-juelich.de/v1/"
LLM_API_MODEL = "alias-fast-experimental"


class LLM:
    """
    A class to handle interactions with a generative Large Language Model (LLM).
    It can rewrite queries, summarize context, and generate final answers.
    """

    def __init__(self, api_key: str, base_url: str, model: str):
        """
        Initializes the OpenAI client for the LLM.
        """
        self.model = model
        self.client = None
        if not api_key:
            print("❌ LLM API_KEY not found in .env file.")
            return

        print("Initializing LLM client...")
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        print("✅ LLM client initialized successfully.")

    def generate_query_pool(self, query: str, num_queries: int = 4) -> list:
        """
        Generates a pool of related search queries from the user's initial question.
        """
        if not self.client:
            print("Cannot generate query pool: Client not initialized.")
            return [query]

        print(f"\n--- Generating a pool of {num_queries} queries for: '{query}' ---")
        system_prompt = (
            "You are a helpful AI research assistant. Your task is to generate a list of "
            f"{num_queries} diverse and relevant search engine queries based on the user's "
            "question. These queries should explore different facets of the original question. "
            "Return the queries as a valid JSON object with a single key 'queries' containing a list of strings. For example: "
            '{"queries": ["query 1", "query 2", "query 3"]}'
        )
        user_prompt = f"User question: \"{query}\""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=0.7,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"}
            )

            response_content = response.choices[0].message.content
            query_list = [query]  # Default to original query

            try:
                # First attempt: direct parsing
                response_data = json.loads(response_content)
                query_list = response_data.get("queries", [query])
            except json.JSONDecodeError:
                # Second attempt (self-correction): find JSON within the string
                print("⚠️ LLM did not return valid JSON. Attempting to repair...")
                match = re.search(r'\{.*\}', response_content, re.DOTALL)
                if match:
                    json_str = match.group(0)
                    try:
                        response_data = json.loads(json_str)
                        query_list = response_data.get("queries", [query])
                        print("✅ Successfully repaired and parsed JSON from response.")
                    except json.JSONDecodeError:
                        print("❌ Repair failed. Could not parse extracted JSON.")
                else:
                    print("❌ Repair failed. No JSON object found in the response.")

            print(f"✅ Generated query pool: {query_list}")
            return query_list

        except Exception as e:
            print(f"❌ An unexpected error occurred while generating the query pool: {e}")
            return [query]

    def rewrite_query(self, query: str) -> str:
        """
        Uses the LLM to rewrite the user's query into a more optimal form for a search engine.
        """
        if not self.client:
            return query

        print(f"\n--- Rewriting query: '{query}' ---")
        system_prompt = (
            "You are an expert at rewriting user questions into high-quality search engine queries. "
            "Convert the user question into a concise, keyword-focused query. "
            "Only provide the rewritten query."
        )
        user_prompt = f"User question: \"{query}\""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=0.0,
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
            )
            rewritten = response.choices[0].message.content.strip().replace("\"", "")
            print(f"✅ Rewritten query: '{rewritten}'")
            return rewritten
        except Exception as e:
            print(f"❌ An error occurred while rewriting the query: {e}")
            return query

    def summarize_context(self, context_df: pd.DataFrame, query: str) -> str:
        """
        Summarizes the retrieved context in relation to the original query.
        """
        if not self.client or context_df.empty:
            return ""

        print(f"--- Summarizing context of size {len(context_df)} for query: '{query}' ---")
        context_texts = "\n- ".join(context_df['text'].tolist())
        system_prompt = (
            "You are a helpful AI assistant. Your task is to synthesize the provided, fragmented "
            "context snippets into a single, short, coherent paragraph. Focus only on the facts "
            "presented in the context that are relevant to the user's question."
        )
        user_prompt = (
            f"Please summarize the key information in the following context regarding the question: '{query}'\n\n"
            f"Context Snippets:\n- {context_texts}"
        )
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=0.2,
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"❌ An error occurred while generating the summary: {e}")
            return ""

    def answer_question_from_summary(self, summarized_context: str, query: str) -> str:
        """
        Generates a final answer from the summarized context.
        """
        if not self.client or not summarized_context:
            return ""
        print(f"--- Generating answer from summary for query: '{query}' ---")
        system_prompt = (
            "You are a helpful AI assistant. Your task is to provide a clear and direct answer in max. 5 sentences "
            "to the user's question using ONLY the provided summary."
        )
        user_prompt = (
            f"Based on the following summary, please answer the question.\n\n"
            f"Summary:\n{summarized_context}\n\n"
            f"Question: {query}"
        )
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=0.1,
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"❌ An error occurred while generating the answer from summary: {e}")
            return ""

    def answer_question_from_context(self, context_df: pd.DataFrame, query: str) -> str:
        """
        Generates a final answer directly from the raw context.
        """
        if not self.client or context_df.empty:
            return ""
        print(f"--- Generating answer from context of size {len(context_df)} for query: '{query}' ---")
        context_texts = "\n- ".join(context_df['text'].tolist())
        system_prompt = (
            "You are a helpful AI assistant. Your task is to provide a clear and direct answer in max. 5 sentences "
            "to the user's question using ONLY the provided context snippets."
        )
        user_prompt = (
            f"Based on the following context snippets, please answer the question.\n\n"
            f"Context Snippets:\n- {context_texts}\n\n"
            f"Question: {query}"
        )
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=0.1,
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"❌ An error occurred while generating the answer from context: {e}")
            return ""


# --- Main Execution Block ---
if __name__ == '__main__':
    test_questions = [
        "What are the health benefits of a Mediterranean diet?", "How does climate change affect coral reefs?",
        "What is the process of photosynthesis?", "Why is the sky blue?", "What are the main causes of inflation?"
    ]  # Shortened list for faster testing

    es_api_key = config.get("ES_API_KEY")
    retriever = SourceRetriever(
        host="https://elasticsearch.bw.webis.de:9200", api_key=es_api_key,
        serps_index="aql_serps", results_index="aql_results"
    )
    llm_generator = LLM(api_key=LLM_API_KEY, base_url=LLM_API_URL, model=LLM_API_MODEL)

    if retriever.es_client and llm_generator.client:
        for i, user_question in enumerate(test_questions):
            print(f"\n\n{'=' * 25} PROCESSING QUESTION {i + 1}/{len(test_questions)} {'=' * 25}")
            print(f"ORIGINAL QUERY: {user_question}")

            # --- Pipeline 1 & 2: Use Original Query ---
            context_dataframe = retriever.get_context(user_question)
            if not context_dataframe.empty:
                print(f"\n\n--- PIPELINE 1&2 CONTEXT (Original Query): {len(context_dataframe)} snippets ---")
                summarized_context = llm_generator.summarize_context(context_dataframe, user_question)
                if summarized_context:
                    final_answer_from_summary = llm_generator.answer_question_from_summary(summarized_context,
                                                                                           user_question)
                    print("\n--- [Pipeline 1] Final Answer (from Summary) ---\n" + final_answer_from_summary)

                final_answer_from_context = llm_generator.answer_question_from_context(context_dataframe, user_question)
                print("\n--- [Pipeline 2] Final Answer (from Context) ---\n" + final_answer_from_context)
            else:
                print("\n--- No context retrieved for original query. ---")

            # --- Pipeline 3: Use Rewritten Query ---
            rewritten_query = llm_generator.rewrite_query(user_question)
            rewritten_context_df = retriever.get_context(rewritten_query)
            if not rewritten_context_df.empty:
                print(f"\n\n--- PIPELINE 3 CONTEXT (Rewritten Query): {len(rewritten_context_df)} snippets ---")
                rewritten_summarized_context = llm_generator.summarize_context(rewritten_context_df, user_question)
                if rewritten_summarized_context:
                    final_answer_rewritten = llm_generator.answer_question_from_summary(rewritten_summarized_context,
                                                                                        user_question)
                    print(
                        "\n--- [Pipeline 3] Final Answer (from Rewritten Query -> Summary) ---\n" + final_answer_rewritten)
            else:
                print("\n--- No context retrieved for rewritten query. ---")

            # --- Pipeline 4: Use Query Pool ---
            query_pool = llm_generator.generate_query_pool(user_question, 25)
            all_contexts = []
            for q in query_pool:
                pooled_df = retriever.get_context(q)
                if not pooled_df.empty:
                    all_contexts.append(pooled_df)

            if all_contexts:
                full_context_df = pd.concat(all_contexts, ignore_index=True).drop_duplicates(
                    subset=['text']).reset_index(drop=True)

                # OPTIMIZATION: Truncate the massive pooled context to avoid API errors.
                # We will take the top 100 most relevant snippets from the combined pool.
                truncated_pool_df = full_context_df.head(100)

                print(
                    f"\n\n--- PIPELINE 4 CONTEXT (Query Pool): {len(full_context_df)} total snippets, using top {len(truncated_pool_df)} ---")
                final_answer_from_pool = llm_generator.answer_question_from_context(truncated_pool_df, user_question)
                print("\n--- [Pipeline 4] Final Answer (from Pooled Context) ---\n" + final_answer_from_pool)
            else:
                print("\n--- No context retrieved for any query in the pool. ---")

            print(f"\n{'=' * 25} FINISHED QUESTION {i + 1}/{len(test_questions)} {'=' * 25}")