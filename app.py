import streamlit as st
import pandas as pd
from SourceRetriever import SourceRetriever
from LLM import LLM
from dotenv import dotenv_values
import random
import subprocess
import sys

# --- Page Configuration ---
st.set_page_config(
    page_title="RAG Pipeline Analysis Tool",
    page_icon="🔬",
    layout="wide"
)

# --- Navigation ---
col1, col2 = st.columns([4, 1])
with col2:
    if st.button("🏆 Go to Ranking Tool", use_container_width=True):
        # Launch the ranking app in a new process
        subprocess.Popen([sys.executable, "-m", "streamlit", "run", "app_rank.py", "--server.port", "8502"])
        st.info("Ranking app öffnet sich in einem neuen Tab auf Port 8502")

# --- Pipeline Functions ---
def run_pipeline_1(query, retriever, llm_generator):
    """Pipeline 1: Original Query (Direct Context → Filter → Answer)"""
    try:
        context_df = retriever.get_context(query)
        original_context = context_df.copy() if not context_df.empty else pd.DataFrame()

        if not context_df.empty:
            # Apply context filtering for Pipeline 1
            filtered_context_df = llm_generator.filter_context(context_df, query)

            if not filtered_context_df.empty:
                answer = llm_generator.answer_question_from_context(filtered_context_df, query)
                return {
                    'answer': answer or "Could not generate an answer.",
                    'original_context': original_context,
                    'filtered_context': filtered_context_df
                }
            else:
                return {
                    'answer': "No relevant context found for this query.",
                    'original_context': original_context,
                    'filtered_context': pd.DataFrame()
                }
        return {
            'answer': "No context retrieved.",
            'original_context': pd.DataFrame(),
            'filtered_context': pd.DataFrame()
        }
    except Exception as e:
        return {
            'answer': f"Error in Pipeline 1: {str(e)}",
            'original_context': pd.DataFrame(),
            'filtered_context': pd.DataFrame()
        }

def run_pipeline_2(query, retriever, llm_generator):
    """Pipeline 2: Original Query (Summary → Answer)"""
    try:
        context_df = retriever.get_context(query)

        if not context_df.empty:
            summarized_context = llm_generator.summarize_context(context_df, query)
            if summarized_context:
                answer = llm_generator.answer_question_from_summary(summarized_context, query)
                return {
                    'answer': answer or "Could not generate an answer.",
                    'context': context_df,
                    'summary': summarized_context
                }
        return {
            'answer': "No context retrieved.",
            'context': pd.DataFrame(),
            'summary': ""
        }
    except Exception as e:
        return {
            'answer': f"Error in Pipeline 2: {str(e)}",
            'context': pd.DataFrame(),
            'summary': ""
        }

def run_pipeline_3(query, retriever, llm_generator):
    """Pipeline 3: Rewritten Query (Rewrite → Summary → Answer)"""
    try:
        rewritten_query = llm_generator.rewrite_query(query)
        context_df = retriever.get_context(rewritten_query)

        if not context_df.empty:
            summarized_context = llm_generator.summarize_context(context_df, query)
            if summarized_context:
                answer = llm_generator.answer_question_from_summary(summarized_context, query)
                return {
                    'answer': answer or "Could not generate an answer.",
                    'rewritten_query': rewritten_query,
                    'context': context_df,
                    'summary': summarized_context
                }
        return {
            'answer': "No context retrieved for rewritten query.",
            'rewritten_query': rewritten_query,
            'context': pd.DataFrame(),
            'summary': ""
        }
    except Exception as e:
        return {
            'answer': f"Error in Pipeline 3: {str(e)}",
            'rewritten_query': query,
            'context': pd.DataFrame(),
            'summary': ""
        }

def run_pipeline_4(query, retriever, llm_generator):
    """Pipeline 4: Query Pool (Query Pool → Filtered Context → Direct Answer)"""
    try:
        query_pool = llm_generator.generate_query_pool(query, 10)
        all_contexts = []
        for q in query_pool:
            pooled_df = retriever.get_context(q)
            if not pooled_df.empty:
                all_contexts.append(pooled_df)

        if all_contexts:
            full_context_df = pd.concat(all_contexts, ignore_index=True).drop_duplicates(
                subset=['text']).reset_index(drop=True)
            truncated_pool_df = full_context_df.head(100)
            original_context = truncated_pool_df.copy()

            # Apply context filtering for Pipeline 4
            filtered_context_df = llm_generator.filter_context(truncated_pool_df, query)

            if not filtered_context_df.empty:
                answer = llm_generator.answer_question_from_context(filtered_context_df, query)
                return {
                    'answer': answer or "Could not generate an answer.",
                    'query_pool': query_pool,
                    'original_context': original_context,
                    'filtered_context': filtered_context_df
                }
            else:
                return {
                    'answer': "No relevant context found for this query.",
                    'query_pool': query_pool,
                    'original_context': original_context,
                    'filtered_context': pd.DataFrame()
                }
        return {
            'answer': "No context retrieved from query pool.",
            'query_pool': query_pool,
            'original_context': pd.DataFrame(),
            'filtered_context': pd.DataFrame()
        }
    except Exception as e:
        return {
            'answer': f"Error in Pipeline 4: {str(e)}",
            'query_pool': [],
            'original_context': pd.DataFrame(),
            'filtered_context': pd.DataFrame()
        }

# --- Load API Keys ---
@st.cache_data
def load_config():
    try:
        config = dotenv_values(".env")
        es_api_key = config.get("ES_API_KEY")
        llm_api_key = config.get("API_KEY")
        if not es_api_key or not llm_api_key:
            st.error(
                "API keys (ES_API_KEY, API_KEY) not found in your .env file. Please make sure it's configured correctly.")
            return None, None
        return es_api_key, llm_api_key
    except Exception as e:
        st.error(f"Could not load the .env file. Please make sure it exists in the root directory. Error: {e}")
        return None, None

ES_API_KEY, LLM_API_KEY = load_config()

if not ES_API_KEY or not LLM_API_KEY:
    st.stop()

# --- Caching the Clients ---
@st.cache_resource
def get_retriever():
    retriever = SourceRetriever(
        host="https://elasticsearch.bw.webis.de:9200",
        api_key=ES_API_KEY,
        serps_index="aql_serps",
        results_index="aql_results"
    )
    if not retriever.es_client:
        st.error("Failed to connect to Elasticsearch. Please check your VPN and API key.")
        return None
    return retriever

@st.cache_resource
def get_llm_generator():
    llm = LLM(
        api_key=LLM_API_KEY,
        base_url="https://api.helmholtz-blablador.fz-juelich.de/v1/",
        model="alias-fast-experimental"
    )
    if not llm.client:
        st.error("Failed to initialize the LLM client. Please check your API key.")
        return None
    return llm

# --- Main App UI ---
st.title("🔬 RAG Pipeline Analysis Tool")
st.markdown(
    """
This application allows you to test and compare different Retrieval-Augmented Generation (RAG) strategies. 
Enter a question below and click 'Run Analysis' to see how each pipeline performs.
"""
)

retriever = get_retriever()
llm_generator = get_llm_generator()

if not retriever or not llm_generator:
    st.warning("One or more services could not be initialized. The app cannot proceed.")
    st.stop()

user_question = st.text_input(
    "Enter your question here:",
    "What are the health benefits of a Mediterranean diet?"
)

if st.button("Run Analysis", type="primary"):
    if not user_question:
        st.warning("Please enter a question.")
    else:
        st.success(f"Processing query: **{user_question}**")

        # --- Pipeline 1: Original Query (Direct Context → Filter → Answer) ---
        st.header("Pipeline 1: Original Query (Direct Context → Filter → Answer)")
        with st.spinner("Running Pipeline 1..."):
            pipeline1_result = run_pipeline_1(user_question, retriever, llm_generator)

        if not pipeline1_result['original_context'].empty:
            st.info(f"Retrieved **{len(pipeline1_result['original_context'])}** snippets, filtered to **{len(pipeline1_result['filtered_context'])}** relevant snippets.")

            col1, col2 = st.columns(2)
            with col1:
                with st.expander("Show Original Context"):
                    st.dataframe(pipeline1_result['original_context'])
            with col2:
                with st.expander("Show Filtered Context"):
                    if not pipeline1_result['filtered_context'].empty:
                        st.dataframe(pipeline1_result['filtered_context'])
                    else:
                        st.info("No relevant context found after filtering.")

            st.subheader("Final Answer")
            st.success(pipeline1_result['answer'])
        else:
            st.warning("No context was retrieved for Pipeline 1.")

        # --- Pipeline 2: Original Query (Summary → Answer) ---
        st.header("Pipeline 2: Original Query (Summary → Answer)")
        with st.spinner("Running Pipeline 2..."):
            pipeline2_result = run_pipeline_2(user_question, retriever, llm_generator)

        if not pipeline2_result['context'].empty:
            st.info(f"Retrieved **{len(pipeline2_result['context'])}** snippets for Pipeline 2.")
            with st.expander("Show Retrieved Context for Pipeline 2"):
                st.dataframe(pipeline2_result['context'])

            with st.expander("Show Pipeline 2 Result"):
                st.subheader("Intermediate Summary")
                st.write(pipeline2_result['summary'] or "Could not generate summary.")
                st.subheader("Final Answer")
                st.success(pipeline2_result['answer'])
        else:
            st.warning("No context was retrieved for Pipeline 2.")

        # --- Pipeline 3: Rewritten Query (Rewrite → Summary → Answer) ---
        st.header("Pipeline 3: Rewritten Query (Rewrite → Summary → Answer)")
        with st.spinner("Running Pipeline 3..."):
            pipeline3_result = run_pipeline_3(user_question, retriever, llm_generator)

        st.subheader("LLM-Rewritten Query")
        st.info(f"**{pipeline3_result['rewritten_query']}**")

        if not pipeline3_result['context'].empty:
            st.info(f"Retrieved **{len(pipeline3_result['context'])}** snippets for Pipeline 3.")
            with st.expander("Show Retrieved Context for Pipeline 3"):
                st.dataframe(pipeline3_result['context'])

            with st.expander("Show Pipeline 3 Result"):
                st.subheader("Intermediate Summary")
                st.write(pipeline3_result['summary'] or "Could not generate summary.")
                st.subheader("Final Answer")
                st.success(pipeline3_result['answer'])
        else:
            st.warning("No context was retrieved for Pipeline 3.")

        # --- Pipeline 4: Query Pool (Query Pool → Filtered Context → Direct Answer) ---
        st.header("Pipeline 4: Query Pool (Query Pool → Filtered Context → Direct Answer)")
        with st.spinner("Running Pipeline 4..."):
            pipeline4_result = run_pipeline_4(user_question, retriever, llm_generator)

        if pipeline4_result['query_pool']:
            st.subheader("Generated Query Pool")
            for i, query in enumerate(pipeline4_result['query_pool'], 1):
                st.write(f"{i}. {query}")

        if not pipeline4_result['original_context'].empty:
            st.info(f"Retrieved **{len(pipeline4_result['original_context'])}** total unique snippets, filtered to **{len(pipeline4_result['filtered_context'])}** relevant snippets.")

            col1, col2 = st.columns(2)
            with col1:
                with st.expander("Show Original Pooled Context"):
                    st.dataframe(pipeline4_result['original_context'])
            with col2:
                with st.expander("Show Filtered Context"):
                    if not pipeline4_result['filtered_context'].empty:
                        st.dataframe(pipeline4_result['filtered_context'])
                    else:
                        st.info("No relevant context found after filtering.")

            st.subheader("Final Answer")
            st.success(pipeline4_result['answer'])
        else:
            st.warning("No context could be retrieved for any query in the pool.")
