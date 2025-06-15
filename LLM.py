import pandas as pd
from dotenv import dotenv_values
from openai import OpenAI
from SourceRetriever import SourceRetriever

# --- Configuration ---
# Loads environment variables from the .env file.
# Make sure your .env file contains the API_KEY for the LLM.
config = dotenv_values(".env")
LLM_API_KEY = config.get("API_KEY")  # Use the key name from the notebook
LLM_API_URL = "https://api.helmholtz-blablador.fz-juelich.de/v1/"
LLM_API_MODEL = "alias-fast"  # Best for fast development runs


class LLM:
    """
    A class to handle interactions with a generative Large Language Model (LLM).
    It can summarize context and generate final answers based on that summary.
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

    def summarize_context(self, context_df: pd.DataFrame, query: str) -> str:
        """
        Summarizes the retrieved context in relation to the original query.
        This is the intermediate step for your thesis.
        """
        if not self.client or context_df.empty:
            if context_df.empty:
                print("Cannot summarize: The provided context DataFrame is empty.")
            return ""

        print(f"\n--- Summarizing {len(context_df)} snippets for query: '{query}' ---")
        context_texts = "\n- ".join(context_df['text'].tolist())

        system_prompt = (
            "You are a helpful AI assistant. Your task is to synthesize the provided, fragmented "
            "context snippets into a single, concise, and neutral summary. Focus only on the facts "
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
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )
            summary = response.choices[0].message.content
            print("✅ Summary generated successfully.")
            return summary.strip()
        except Exception as e:
            print(f"❌ An error occurred while generating the summary: {e}")
            return ""

    def answer_question_from_summary(self, summarized_context: str, query: str) -> str:
        """
        Generates a final answer to the user's question based on the summarized context.
        """
        if not self.client or not summarized_context:
            print("Cannot answer question: Client not initialized or summarized context is empty.")
            return ""

        print(f"\n--- Generating final answer from SUMMARY for query: '{query}' ---")
        system_prompt = (
            "You are a helpful AI assistant. Your task is to provide a clear, short and direct in max. 5 sentences "
            "to the user's question using ONLY the provided summary. Do not use any outside knowledge."
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
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )
            answer = response.choices[0].message.content
            print("✅ Final answer from summary generated successfully.")
            return answer.strip()
        except Exception as e:
            print(f"❌ An error occurred while generating the final answer from summary: {e}")
            return ""

    def answer_question_from_context(self, context_df: pd.DataFrame, query: str) -> str:
        """
        Generates a final answer directly from the raw context snippets.
        This is the baseline pipeline without the summarization step.
        """
        if not self.client or context_df.empty:
            print("Cannot answer question: Client not initialized or context is empty.")
            return ""

        print(f"\n--- Generating final answer from RAW CONTEXT for query: '{query}' ---")
        context_texts = "\n- ".join(context_df['text'].tolist())

        system_prompt = (
            "You are a helpful AI assistant. Your task is to provide a clear, short and direct in max. 5 sentences "
            "to the user's question using ONLY the provided context snippets. Do not use any outside knowledge."
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
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )
            answer = response.choices[0].message.content
            print("✅ Final answer from raw context generated successfully.")
            return answer.strip()
        except Exception as e:
            print(f"❌ An error occurred while generating the final answer from raw context: {e}")
            return ""


# --- Usage Example ---
if __name__ == '__main__':
    test_questions = [
        "What are the health benefits of a Mediterranean diet?", "How does climate change affect coral reefs?",
        "What is the process of photosynthesis?", "Why is the sky blue?", "What are the main causes of inflation?",
        "How do vaccines work?", "What is the history of the internet?", "What are black holes?",
        "How does solar power work?", "What is the difference between nuclear fission and fusion?", "Why do we dream?",
        "What are the symptoms of long COVID?", "How to bake a sourdough bread?",
        "What are the arguments for and against universal basic income?",
        "What is the significance of the Rosetta Stone?",
        "How does a 3D printer work?", "What are the ethical implications of artificial intelligence?",
        "Why is biodiversity important for ecosystems?", "What was the impact of the printing press?",
        "How to learn a new language effectively?", "What is blockchain technology?",
        "What are the causes of the Roman Empire's fall?",
        "How does quantum computing differ from classical computing?",
        "What are the benefits of mindfulness and meditation?", "Why is sleep important for health?",
        "What is the theory of relativity?", "How are electric car batteries recycled?", "What is the gig economy?",
        "What are the main principles of stoicism?", "How does GPS technology work?"
    ]

    es_api_key = config.get("ES_API_KEY")
    retriever = SourceRetriever(
        host="https://elasticsearch.bw.webis.de:9200", api_key=es_api_key,
        serps_index="aql_serps", results_index="aql_results"
    )

    llm_generator = LLM(
        api_key=LLM_API_KEY, base_url=LLM_API_URL, model=LLM_API_MODEL
    )

    if retriever.es_client and llm_generator.client:
        for i, user_question in enumerate(test_questions):
            print(f"\n\n{'=' * 20} PROCESSING QUESTION {i + 1}/{len(test_questions)} {'=' * 20}")
            print(f"QUERY: {user_question}")

            # Step 1: Get the raw context (same for both pipelines)
            context_dataframe = retriever.get_context(user_question)

            if not context_dataframe.empty:
                # --- Pipeline 1: With Summarization Step ---
                print("\n\n--- PIPELINE 1: WITH SUMMARIZATION ---")
                summarized_context = llm_generator.summarize_context(context_dataframe, user_question)
                if summarized_context:
                    final_answer_from_summary = llm_generator.answer_question_from_summary(summarized_context, user_question)
                    if final_answer_from_summary:
                        print("\n--- Final Answer (from Summary) ---")
                        print(final_answer_from_summary)
                        print("---------------------------------")

                # --- Pipeline 2: Direct from Raw Context ---
                print("\n\n--- PIPELINE 2: DIRECT FROM CONTEXT ---")
                final_answer_from_context = llm_generator.answer_question_from_context(context_dataframe, user_question)
                if final_answer_from_context:
                    print("\n--- Final Answer (from Context) ---")
                    print(final_answer_from_context)
                    print("---------------------------------")

            else:
                print("\n--- No context was retrieved, skipping pipelines. ---")
            print(f"{'=' * 20} FINISHED QUESTION {i + 1}/{len(test_questions)} {'=' * 20}")

