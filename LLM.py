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
           "You are a helpful AI research assistant specialized in query expansion for search engines. "
           "Your task is to generate multiple relevant search queries from the user's input to improve search coverage.\n\n"
           "Instructions:\n"
           "1. Take the user's input query and identify meaningful terms (ignore stop words like 'the', 'a', 'an')\n"
           "2. Generate strategic combinations of these terms that would be useful for search\n"
           "3. Remove question words (what, how, where, when, why, who) from combinations\n"
           "4. Exclude overly generic terms that don't add search value (like 'best', 'good', 'how to')\n"
           "5. Always include the original query as the first item\n"
           "Return ONLY a valid JSON object with this exact format:\n"
           '{"queries": ["original query", "variation 1", "variation 2", ...]}\n\n'
           "Example:\n"
           "Input: 'What is the best carbonara recipe?'\n"
           "Output: "
           '{"queries": ["What is the best carbonara recipe?", "carbonara recipe", "carbonara ingredients", "authentic carbonara", "carbonara cooking method"]}\n\n'
           "Focus on creating queries that would find different but relevant information about the topic."
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
                match = re.search(r'\{.*}', response_content, re.DOTALL)
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
            "to the user's question using ONLY the provided context snippets"
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

    def filter_context(self, context_df: pd.DataFrame, query: str) -> pd.DataFrame:
        """
        Filters the context by removing snippets that are not relevant to the user's query.
        Uses the LLM to evaluate each context snippet for relevance.
        Returns empty DataFrame if no relevant context is found.
        """
        if not self.client or context_df.empty:
            return context_df

        print(f"--- Filtering context of size {len(context_df)} for query: '{query}' ---")

        filtered_contexts = []

        # Process contexts in batches to avoid token limits
        batch_size = 5
        for i in range(0, len(context_df), batch_size):
            batch = context_df.iloc[i:i + batch_size]
            batch_texts = []
            for idx, row in batch.iterrows():
                batch_texts.append(f"[{idx}]: {row['text']}")

            context_batch = "\n".join(batch_texts)

            system_prompt = (
                "You are an expert at evaluating text relevance for question-answering systems. Your task is to determine which context snippets "
                "are relevant to answering the user's question with high precision and recall.\n\n"
                "Evaluation criteria:\n"
                "- Include snippets that directly answer the question or provide key information\n"
                "- Include snippets with relevant facts, data, or examples related to the topic\n"
                "- Include snippets that provide necessary background or context\n"
                "- Exclude snippets that are only tangentially related or off-topic\n"
                "- Exclude snippets that contain no useful information for answering the question\n\n"
                "Return ONLY a JSON object with a 'relevant_indices' key containing a list of indices (numbers) of the relevant snippets. "
                "If NO snippets are relevant, return an empty list.\n"
                "Examples:\n"
                "- Relevant snippets found: {\"relevant_indices\": [0, 2, 4]}\n"
                "- No relevant snippets: {\"relevant_indices\": []}\n"
                "- Single relevant snippet: {\"relevant_indices\": [1]}"
            )

            user_prompt = (
                f"Question: {query}\n\n"
                f"Context snippets:\n{context_batch}\n\n"
                f"Which of these snippets (by their index numbers) are relevant to answering the question? "
                f"Only include snippets that contain information that could help answer the question. "
                f"Be strict - if a snippet doesn't directly help answer the question, don't include it."
            )

            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    temperature=0.0,
                    messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
                )

                response_text = response.choices[0].message.content.strip()

                # Parse JSON response
                try:
                    result = json.loads(response_text)
                    relevant_indices = result.get('relevant_indices', [])

                    # Add relevant contexts from this batch
                    for idx in relevant_indices:
                        if idx < len(batch):
                            filtered_contexts.append(batch.iloc[idx])

                except (json.JSONDecodeError, KeyError) as e:
                    print(f"⚠️ Could not parse LLM response for filtering batch {i // batch_size + 1}: {e}")
                    # If parsing fails, include all contexts from this batch as fallback
                    filtered_contexts.extend([batch.iloc[j] for j in range(len(batch))])

            except Exception as e:
                print(f"❌ Error occurred while filtering context batch {i // batch_size + 1}: {e}")
                # If API call fails, include all contexts from this batch as fallback
                filtered_contexts.extend([batch.iloc[j] for j in range(len(batch))])

        if filtered_contexts:
            filtered_df = pd.DataFrame(filtered_contexts).reset_index(drop=True)
            print(f"✅ Context filtered: {len(context_df)} → {len(filtered_df)} snippets")
            return filtered_df
        else:
            print("⚠️ No relevant context found after filtering - returning empty DataFrame")
            return pd.DataFrame()  # Return empty DataFrame instead of original context