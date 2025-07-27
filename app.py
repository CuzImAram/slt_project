import streamlit as st
import pandas as pd
from SourceRetriever import SourceRetriever
from LLM import LLM
from dotenv import dotenv_values
import random
import io

# --- Page Configuration ---
st.set_page_config(
    page_title="RAG Pipeline Comparison Tool",
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

# --- Test Queries ---
TEST_QUERIES = [
    "How does climate change affect coral reefs?"
]

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
            'answer': "No context retrieved from query pool.",
            'original_context': pd.DataFrame(),
            'filtered_context': pd.DataFrame()
        }
    except Exception as e:
        return {
            'answer': f"Error in Pipeline 4: {str(e)}",
            'original_context': pd.DataFrame(),
            'filtered_context': pd.DataFrame()
        }

# --- Initialize Session State ---
if 'results_ready' not in st.session_state:
    st.session_state.results_ready = False
if 'comparison_results' not in st.session_state:
    st.session_state.comparison_results = []
if 'current_vote_index' not in st.session_state:
    st.session_state.current_vote_index = 0
if 'votes' not in st.session_state:
    st.session_state.votes = []
if 'voting_complete' not in st.session_state:
    st.session_state.voting_complete = False

# --- Main App UI ---
st.title("🔬 RAG Pipeline Comparison Tool")
st.markdown(
    """
This application compares Pipeline 1 (Direct Context → Answer) vs Pipeline 4 (Query Pool → Direct Answer).
Click 'Start Comparison' to run 15 test queries through both pipelines and then vote on the results.
"""
)

retriever = get_retriever()
llm_generator = get_llm_generator()

if not retriever or not llm_generator:
    st.warning("One or more services could not be initialized. The app cannot proceed.")
    st.stop()

# --- Run Comparison ---
if not st.session_state.results_ready:
    if st.button("Start Comparison", type="primary"):
        st.session_state.comparison_results = []

        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()

        for i, query in enumerate(TEST_QUERIES):
            status_text.text(f"Processing query {i+1}/15: {query[:50]}...")

            # Run both pipelines
            pipeline1_result = run_pipeline_1(query, retriever, llm_generator)
            pipeline4_result = run_pipeline_4(query, retriever, llm_generator)

            # Randomly assign which pipeline goes to left/right
            if random.choice([True, False]):
                left_result = ("Pipeline 1", pipeline1_result)
                right_result = ("Pipeline 4", pipeline4_result)
            else:
                left_result = ("Pipeline 4", pipeline4_result)
                right_result = ("Pipeline 1", pipeline1_result)

            st.session_state.comparison_results.append({
                'query': query,
                'left': left_result,
                'right': right_result,
                'pipeline1_answer': pipeline1_result['answer'],
                'pipeline4_answer': pipeline4_result['answer']
            })

            progress_bar.progress((i + 1) / len(TEST_QUERIES))

        status_text.text("✅ All queries processed! Ready for voting.")
        st.session_state.results_ready = True
        st.rerun()

# --- Voting Interface ---
elif st.session_state.results_ready and not st.session_state.voting_complete:
    st.header("🗳️ Vote for the Better Answer")

    current_idx = st.session_state.current_vote_index
    if current_idx < len(st.session_state.comparison_results):
        result = st.session_state.comparison_results[current_idx]

        # Progress indicator
        st.progress((current_idx + 1) / len(st.session_state.comparison_results))
        st.text(f"Question {current_idx + 1} of {len(st.session_state.comparison_results)}")

        # Display query
        st.subheader("Query:")
        st.info(result['query'])

        # Display answers side by side
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Answer A")
            st.write(result['left'][1]['answer'])

        with col2:
            st.subheader("Answer B")
            st.write(result['right'][1]['answer'])

        # Context Viewing Dropdown
        st.markdown("---")
        st.subheader("📄 Context Inspection")

        # Get pipeline results for context display
        pipeline1_result = None
        pipeline4_result = None

        # Determine which pipeline is which
        for name, result_data in [result['left'], result['right']]:
            if name == "Pipeline 1":
                pipeline1_result = result_data
            elif name == "Pipeline 4":
                pipeline4_result = result_data

        # Create context view options
        context_options = []
        context_data = {}

        if pipeline1_result is not None:
            if not pipeline1_result['original_context'].empty:
                context_options.append("Pipeline 1 - Original Context")
                context_data["Pipeline 1 - Original Context"] = pipeline1_result['original_context']

            if not pipeline1_result['filtered_context'].empty:
                context_options.append("Pipeline 1 - Filtered Context")
                context_data["Pipeline 1 - Filtered Context"] = pipeline1_result['filtered_context']

        if pipeline4_result is not None:
            if not pipeline4_result['original_context'].empty:
                context_options.append("Pipeline 4 - Original Context")
                context_data["Pipeline 4 - Original Context"] = pipeline4_result['original_context']

            if not pipeline4_result['filtered_context'].empty:
                context_options.append("Pipeline 4 - Filtered Context")
                context_data["Pipeline 4 - Filtered Context"] = pipeline4_result['filtered_context']

        if context_options:
            selected_context = st.selectbox(
                "Select context to view:",
                options=["None"] + context_options,
                index=0
            )

            if selected_context != "None" and selected_context in context_data:
                selected_df = context_data[selected_context]

                # Display context statistics
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Number of snippets", len(selected_df))
                with col2:
                    if "Pipeline 1" in selected_context and pipeline1_result is not None:
                        original_count = len(pipeline1_result['original_context'])
                        filtered_count = len(pipeline1_result['filtered_context'])
                        if "Filtered" in selected_context and original_count > 0:
                            retention_rate = (filtered_count / original_count) * 100
                            st.metric("Retention rate", f"{retention_rate:.1f}%")
                    elif "Pipeline 4" in selected_context and pipeline4_result is not None:
                        original_count = len(pipeline4_result['original_context'])
                        filtered_count = len(pipeline4_result['filtered_context'])
                        if "Filtered" in selected_context and original_count > 0:
                            retention_rate = (filtered_count / original_count) * 100
                            st.metric("Retention rate", f"{retention_rate:.1f}%")

                # Display context snippets
                with st.expander(f"View {selected_context} snippets", expanded=True):
                    for idx, row in selected_df.iterrows():
                        st.markdown(f"**Snippet {idx + 1}:**")
                        st.text(row['text'])
                        st.markdown("---")
        else:
            st.info("No context available for this query.")

        # Voting buttons
        st.markdown("---")
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("👈 Answer A is Better", use_container_width=True):
                winner = result['left'][0]  # Pipeline name
                st.session_state.votes.append({
                    'query': result['query'],
                    'output_1': result['pipeline1_answer'],
                    'output_2': result['pipeline4_answer'],
                    'winner': winner
                })
                st.session_state.current_vote_index += 1
                st.rerun()

        with col2:
            if st.button("🤷 Don't Care", use_container_width=True):
                st.session_state.votes.append({
                    'query': result['query'],
                    'output_1': result['pipeline1_answer'],
                    'output_2': result['pipeline4_answer'],
                    'winner': 'Don\'t Care'
                })
                st.session_state.current_vote_index += 1
                st.rerun()

        with col3:
            if st.button("👉 Answer B is Better", use_container_width=True):
                winner = result['right'][0]  # Pipeline name
                st.session_state.votes.append({
                    'query': result['query'],
                    'output_1': result['pipeline1_answer'],
                    'output_2': result['pipeline4_answer'],
                    'winner': winner
                })
                st.session_state.current_vote_index += 1
                st.rerun()

    else:
        st.session_state.voting_complete = True
        st.rerun()

# --- Results and Export ---
elif st.session_state.voting_complete:
    st.header("🎉 Voting Complete!")

    # Display results summary
    votes_df = pd.DataFrame(st.session_state.votes)

    st.subheader("Results Summary")
    pipeline1_wins = len(votes_df[votes_df['winner'] == 'Pipeline 1'])
    pipeline4_wins = len(votes_df[votes_df['winner'] == 'Pipeline 4'])
    dont_care = len(votes_df[votes_df['winner'] == 'Don\'t Care'])

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Pipeline 1 Wins", pipeline1_wins)
    with col2:
        st.metric("Pipeline 4 Wins", pipeline4_wins)
    with col3:
        st.metric("Don't Care", dont_care)

    # Export functionality
    st.subheader("Export Results")

    # Create CSV content
    csv_buffer = io.StringIO()
    votes_df.to_csv(csv_buffer, index=False)
    csv_content = csv_buffer.getvalue()

    st.download_button(
        label="📥 Download Results as CSV",
        data=csv_content,
        file_name="pipeline_comparison_results.csv",
        mime="text/csv"
    )

    # Display detailed results
    with st.expander("View Detailed Results"):
        st.dataframe(votes_df)

    # Reset button
    if st.button("🔄 Start New Comparison"):
        st.session_state.results_ready = False
        st.session_state.comparison_results = []
        st.session_state.current_vote_index = 0
        st.session_state.votes = []
        st.session_state.voting_complete = False
        st.rerun()
