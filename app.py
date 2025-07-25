import streamlit as st
import pandas as pd
from SourceRetriever import SourceRetriever
from LLM import LLM
from dotenv import dotenv_values

# --- Page Configuration ---
st.set_page_config(
    page_title="RAG Pipeline Analysis Tool",
    page_icon="🔬",
    layout="wide"
)

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
        model="alias-fast"
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

        # --- Pipeline 1: Original Query (Direct Context → Answer) ---
        st.header("Pipeline 1: Original Query (Direct Context → Answer)")
        with st.spinner("Retrieving context for Pipeline 1..."):
            context_df2 = retriever.get_context(user_question)

        if not context_df2.empty:
            st.info(f"Retrieved **{len(context_df2)}** snippets for Pipeline 1.")
            with st.expander("Show Retrieved Context for Pipeline 1"):
                st.dataframe(context_df2)

            with st.expander("Show Pipeline 1 Result"):
                with st.spinner("Pipeline 1: Generating answer directly from context..."):
                    answer_p2 = llm_generator.answer_question_from_context(context_df2, user_question)
                st.subheader("Final Answer")
                st.success(answer_p2 or "Could not generate an answer.")
        else:
            st.warning("No context was retrieved for Pipeline 1.")

        # --- Pipeline 2: Original Query (Summary → Answer) ---
        st.header("Pipeline 2: Original Query (Summary → Answer)")
        with st.spinner("Retrieving context for Pipeline 2..."):
            context_df = retriever.get_context(user_question)

        if not context_df.empty:
            st.info(f"Retrieved **{len(context_df)}** snippets for Pipeline 2.")
            with st.expander("Show Retrieved Context for Pipeline 2"):
                st.dataframe(context_df)

            with st.expander("Show Pipeline 2 Result"):
                with st.spinner("Pipeline 2: Summarizing context and generating answer..."):
                    summarized_context = llm_generator.summarize_context(context_df, user_question)
                    st.subheader("Intermediate Summary")
                    st.write(summarized_context or "Could not generate summary.")

                    if summarized_context:
                        answer_p1 = llm_generator.answer_question_from_summary(summarized_context, user_question)
                        st.subheader("Final Answer")
                        st.success(answer_p1 or "Could not generate an answer.")
        else:
            st.warning("No context was retrieved for Pipeline 2.")

        # --- Pipeline 3: Rewritten Query (Rewrite → Summary → Answer) ---
        st.header("Pipeline 3: Rewritten Query (Rewrite → Summary → Answer)")
        # Rewrite Query
        with st.spinner("Pipeline 3: Rewriting query..."):
            rewritten_query = llm_generator.rewrite_query(user_question)
        st.subheader("LLM-Rewritten Query")
        st.info(f"**{rewritten_query}**")

        # Retrieve Context
        with st.spinner("Retrieving context for rewritten query..."):
            rewritten_context_df = retriever.get_context(rewritten_query)

        if not rewritten_context_df.empty:
            st.info(f"Retrieved **{len(rewritten_context_df)}** snippets for Pipeline 3.")
            with st.expander("Show Retrieved Context for Pipeline 3"):
                st.dataframe(rewritten_context_df)

            # Generate Summary & Answer
            with st.expander("Show Pipeline 3 Result"):
                with st.spinner("Pipeline 3: Summarizing and generating final answer..."):
                    summarized_context_p3 = llm_generator.summarize_context(rewritten_context_df, user_question)
                    st.subheader("Intermediate Summary")
                    st.write(summarized_context_p3 or "Could not generate summary.")

                    if summarized_context_p3:
                        answer_p3 = llm_generator.answer_question_from_summary(summarized_context_p3, user_question)
                        st.subheader("Final Answer")
                        st.success(answer_p3 or "Could not generate an answer.")
        else:
            st.warning("No context was retrieved for Pipeline 3.")

        # --- Pipeline 4: Query Pool (Query Pool → Direct Answer) ---
        st.header("Pipeline 4: Query Pool (Query Pool → Direct Answer)")
        with st.expander("Show Pipeline 4 Result"):
            with st.spinner("Pipeline 4: Generating query pool..."):
                query_pool = llm_generator.generate_query_pool(user_question, 25)

            with st.expander("Show Generated Query Pool"):
                st.info(query_pool)

            all_contexts = []
            with st.spinner("Pipeline 4: Retrieving context for all pooled queries..."):
                for q in query_pool:
                    pooled_df = retriever.get_context(q)
                    if not pooled_df.empty:
                        all_contexts.append(pooled_df)

            if all_contexts:
                full_context_df = pd.concat(all_contexts, ignore_index=True).drop_duplicates(
                    subset=['text']).reset_index(drop=True)
                truncated_pool_df = full_context_df.head(100)
                st.write(
                    f"Retrieved **{len(full_context_df)}** total unique snippets, using top **{len(truncated_pool_df)}** for answer generation.")

                with st.expander("Show Pooled & Truncated Context for Pipeline 4"):
                    st.dataframe(truncated_pool_df)

                with st.spinner("Pipeline 4: Generating answer from pooled context..."):
                    answer_p4 = llm_generator.answer_question_from_context(truncated_pool_df, user_question)
                st.subheader("Final Answer")
                st.success(answer_p4 or "Could not generate an answer.")
            else:
                st.warning("No context could be retrieved for any query in the pool.")