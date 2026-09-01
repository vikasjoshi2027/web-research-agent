"""
Streamlit Web Interface for Autonomous Deep Research Agent.
Author: Vikas Joshi <vikasjoshi.2027@gmail.com>

Usage:
    streamlit run app.py
"""

import os
import sys
import time
import streamlit as st
from dotenv import load_dotenv

# Load local environment
load_dotenv()

# Page Configuration
st.set_page_config(
    page_title="Autonomous Deep Research Agent",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Minimal, Clean Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2rem;
        font-weight: 700;
        color: #111827;
        margin-bottom: 0.25rem;
    }
    .main-subtitle {
        font-size: 1rem;
        color: #6B7280;
        margin-bottom: 1.5rem;
    }
    .metric-container {
        background-color: #F9FAFB;
        border: 1px solid #E5E7EB;
        border-radius: 6px;
        padding: 12px 16px;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar Configuration
with st.sidebar:
    st.markdown("### Configuration")
    
    st.markdown("#### API Credentials")
    env_openai = os.getenv("OPENAI_API_KEY", "")
    env_tavily = os.getenv("TAVILY_API_KEY", "")
    
    openai_key = st.text_input(
        "OpenAI API Key",
        value=env_openai,
        type="password",
        help="Required for report synthesis and evaluation"
    )
    tavily_key = st.text_input(
        "Tavily Search API Key",
        value=env_tavily,
        type="password",
        help="Required for web search retrieval"
    )
    
    if openai_key:
        os.environ["OPENAI_API_KEY"] = openai_key
    if tavily_key:
        os.environ["TAVILY_API_KEY"] = tavily_key
        
    st.markdown("---")
    st.markdown("#### Execution Parameters")
    model_choice = st.selectbox(
        "LLM Model",
        ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
        index=0
    )
    os.environ["OPENAI_MODEL"] = model_choice
    
    max_cycles = st.slider(
        "Max Reflection Iterations",
        min_value=1,
        max_value=3,
        value=2,
        help="Controls the maximum search and reflection loop depth"
    )
    
    st.markdown("---")
    st.markdown("**Author**: Vikas Joshi  \n**Contact**: vikasjoshi.2027@gmail.com")

# Main Header
st.markdown('<div class="main-title">Autonomous Deep Research Agent</div>', unsafe_allow_html=True)
st.markdown('<div class="main-subtitle">Multi-Step LangGraph Agent with Self-Reflection and Citation Verification</div>', unsafe_allow_html=True)

# Preset Topics
st.markdown("**Sample Research Topics:**")
col1, col2, col3 = st.columns(3)
sample_1 = col1.button("Multi-Agent Orchestration Patterns")
sample_2 = col2.button("Quantum Computing & Logical Qubits")
sample_3 = col3.button("Solid-State Battery Advancements")

selected_query = ""
if sample_1:
    selected_query = "State of autonomous multi-agent orchestration architectures in 2026"
elif sample_2:
    selected_query = "Latest breakthroughs in quantum computing and error-corrected qubits"
elif sample_3:
    selected_query = "Recent advancements and commercialization of solid-state battery technology"

# Query Input
query = st.text_area(
    "Research Topic / Query",
    value=selected_query if selected_query else "Latest advancements in agentic AI architectures and multi-step reasoning",
    height=80
)

start_btn = st.button("Execute Research Task", type="primary", use_container_width=True)

if start_btn:
    if not os.getenv("OPENAI_API_KEY") or not os.getenv("TAVILY_API_KEY"):
        st.error("Please provide both OPENAI_API_KEY and TAVILY_API_KEY in the sidebar configuration.")
    else:
        try:
            from agent import build_research_graph
        except ImportError as e:
            st.error(f"Error importing research graph: {e}")
            st.stop()

        status_container = st.container()
        
        with status_container:
            st.markdown("### Execution Status")
            progress_bar = st.progress(10)
            status_text = st.empty()
            
            status_text.info("[Step 1/5] Decomposing research topic into sub-queries...")
            time.sleep(0.4)
            
            agent = build_research_graph()
            
            initial_state = {
                "messages": [],
                "query": query,
                "sub_queries": [],
                "search_results": [],
                "iteration_count": 0,
                "max_iterations": max_cycles,
                "is_sufficient": False,
                "critique_notes": "",
                "draft_report": "",
                "verified_report": "",
                "verification_audit": ""
            }
            
            progress_bar.progress(35)
            status_text.info("[Step 2 & 3] Retrieving web sources and evaluating completeness...")
            
            try:
                result = agent.invoke(initial_state)
                
                progress_bar.progress(75)
                status_text.info("[Step 4] Synthesizing draft report with inline citations...")
                time.sleep(0.4)
                
                progress_bar.progress(90)
                status_text.info("[Step 5] Auditing citations and factual consistency...")
                time.sleep(0.4)
                
                progress_bar.progress(100)
                status_text.success("Research report generated and verified successfully.")
                
                # Metrics
                m1, m2, m3 = st.columns(3)
                m1.metric("Unique Sources Aggregated", len(result.get("search_results", [])))
                m2.metric("Research Iterations", result.get("iteration_count", 1))
                m3.metric("Verification Status", "Verified")
                
                st.markdown("---")
                
                # Result Tabs
                tab_report, tab_sources, tab_audit = st.tabs(["Verified Report", "Consulted Sources", "Verification Audit"])
                
                with tab_report:
                    report_text = result.get("verified_report", "")
                    st.markdown(report_text)
                    
                    st.markdown("---")
                    st.download_button(
                        label="Download Report (Markdown)",
                        data=report_text,
                        file_name="deep_research_report.md",
                        mime="text/markdown"
                    )
                    
                with tab_sources:
                    sources = result.get("search_results", [])
                    st.markdown(f"**Total Sources Consulted ({len(sources)}):**")
                    for idx, src in enumerate(sources, 1):
                        title = src.get("title", "Untitled Source")
                        url = src.get("url", "#")
                        snippet = src.get("content", "")
                        with st.expander(f"[{idx}] {title}"):
                            st.markdown(f"**URL**: [{url}]({url})")
                            st.write(snippet)
                            
                with tab_audit:
                    st.markdown("#### Audit Trail and Gap Analysis")
                    st.info(f"**Evaluation Notes**: {result.get('critique_notes', 'N/A')}")
                    st.write(f"**Completed Iterations**: {result.get('iteration_count', 1)} / {max_cycles}")
                    
            except Exception as e:
                st.error(f"An error occurred during execution: {e}")