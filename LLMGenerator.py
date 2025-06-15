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


class LLMGenerator:
    """
    A class to handle interactions with a generative Large Language Model (LLM).
    It takes context from the SourceRetriever and generates summaries or answers.
    """

    def __init__(self, api_key: str, base_url: str, model: str):
        """
        Initializes the OpenAI client for the LLM.

        Args:
            api_key (str): The API key for the LLM service.
            base_url (str): The base URL for the LLM API endpoint.
            model (str): The specific model to use for generation.
        """
        self.model = model
        self.client = None
        if not api_key:
            print("❌ LLM API_KEY not found in .env file.")
            return

        print("Initializing LLM client...")
        # Initialize the client, as shown in the 02-llms-solved.ipynb notebook
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        print("✅ LLM client initialized successfully.")

    def summarize_context(self, context_df: pd.DataFrame, query: str) -> str:
        """
        Summarizes the retrieved context in relation to the original query.
        This method is the "Zwischenschritt" (intermediate step) for your thesis.

        Args:
            context_df (pd.DataFrame): The DataFrame containing the 'text' column of snippets.
            query (str): The original user query to focus the summary.

        Returns:
            str: A single, concise summary of the context, or an empty string if an error occurs.
        """
        if not self.client or context_df.empty:
            if context_df.empty:
                print("Cannot summarize: The provided context DataFrame is empty.")
            return ""

        print(f"\n--- Summarizing {len(context_df)} snippets for query: '{query}' ---")

        # 1. Format the context for the prompt
        # We will join all snippet texts into a single block.
        context_texts = "\n- ".join(context_df['text'].tolist())

        # 2. Define the prompts based on notebook examples
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
            # 3. Call the LLM API
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=0.2,  # Lower temperature for more factual, less creative summaries
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


# --- Usage Example ---
if __name__ == '__main__':
    test_questions = [
        "What are the health benefits of a Mediterranean diet?",
        "How does climate change affect coral reefs?",
        "What is the process of photosynthesis?",
        "Why is the sky blue?",
        "What are the main causes of inflation?",
        "How do vaccines work?",
        "What is the history of the internet?",
        "What are black holes?",
        "How does solar power work?",
        "What is the difference between nuclear fission and fusion?",
        "Why do we dream?",
        "What are the symptoms of long COVID?",
        "How to bake a sourdough bread?",
        "What are the arguments for and against universal basic income?",
        "What is the significance of the Rosetta Stone?",
        "How does a 3D printer work?",
        "What are the ethical implications of artificial intelligence?",
        "Why is biodiversity important for ecosystems?",
        "What was the impact of the printing press?",
        "How to learn a new language effectively?",
        "What is blockchain technology?",
        "What are the causes of the Roman Empire's fall?",
        "How does quantum computing differ from classical computing?",
        "What are the benefits of mindfulness and meditation?",
        "Why is sleep important for health?",
        "What is the theory of relativity?",
        "How are electric car batteries recycled?",
        "What is the gig economy?",
        "What are the main principles of stoicism?",
        "How does GPS technology work?"
    ]

    # 1. Initialize the SourceRetriever (from the other file)
    es_api_key = config.get("ES_API_KEY")
    retriever = SourceRetriever(
        host="https://elasticsearch.bw.webis.de:9200",
        api_key=es_api_key,
        serps_index="aql_serps",
        results_index="aql_results"
    )

    # 2. Initialize the LLMGenerator
    llm_generator = LLMGenerator(
        api_key=LLM_API_KEY,
        base_url=LLM_API_URL,
        model=LLM_API_MODEL
    )

    # 3. Proceed only if both clients were initialized successfully
    if retriever.es_client and llm_generator.client:
        # Loop through each question in the test list
        for i, user_question in enumerate(test_questions):
            print(f"\n\n{'=' * 20} PROCESSING QUESTION {i + 1}/{len(test_questions)} {'=' * 20}")
            print(f"QUERY: {user_question}")

            # Step A: Get the context
            context_dataframe = retriever.get_context(user_question)

            if not context_dataframe.empty:
                # Step B: Pass the context to the LLM for summarization
                summarized_context = llm_generator.summarize_context(context_dataframe, user_question)

                if summarized_context:
                    print("\n--- LLM-Generated Summary of Context ---")
                    print(summarized_context)
                    print("----------------------------------------")
            else:
                print("\n--- No context was retrieved, skipping summarization. ---")
            print(f"{'=' * 20} FINISHED QUESTION {i + 1}/{len(test_questions)} {'=' * 20}")

