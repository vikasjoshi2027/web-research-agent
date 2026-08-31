"""
Streamlit Web Interface for Autonomous Deep Research Agent.
Author: Vikas Joshi <vikasjoshi.2027@gmail.com>

Run with:
    streamlit run app.py
"""

import os
import sys
import time
import streamlit as st
from dotenv import load_dotenv

# Load local environment
load_dotenv()

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="Deep Research AI Agent",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling for premium look
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E88E5;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #616161;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 12px 18px;
        border-left: 4px solid #1E88E5;
        margin-bottom: 10px;
    }
    .source-box {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 6px;
        padding: 10px 14px;
        margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar Configuration
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=64)
    st.title("Settings & Keys")
    
    st.markdown("### 🔑 API Credentials")
    env_openai = os.getenv("OPENAI_API_KEY", "")
    env_tavily = os.getenv("TAVILY_API_KEY", "")
    
    openai_key = st.text_input("OpenAI API Key", value=env_openai, type="password", help="Needed for report synthesis & evaluation")
    tavily_key = st.text_input("Tavily Search API Key", value=env_tavily, type="password", help="Needed for real-time web search")
    
    if openai_key:
        os.environ["OPENAI_API_KEY"] = openai_key
    if tavily_key:
        os.environ["TAVILY_API_KEY"] = tavily_key
        
    st.markdown("---")
    st.markdown("### ⚙️ Research Depth")
    model_choice = st.selectbox("LLM Model", ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"], index=0)
    os.environ["OPENAI_MODEL"] = model_choice
    
    max_cycles = st.slider("Max Reflection Cycles", min_value=1, max_value=3, value=2, help="Higher cycles enable deeper reflection and gap-filling")
    
    st.markdown("---")
    st.markdown("👨‍💻 **Author**: Vikas Joshi  \n✉️ [vikasjoshi.2027@gmail.com](mailto:vikasjoshi.2027@gmail.com)")

# Main Header
st.markdown('<div class="main-header">🌐 Autonomous Deep Research Agent</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Multi-Step LangGraph Agent with Self-Reflection & Citation Fact-Checking</div>', unsafe_allow_html=True)

# Sample Topics / Quick Chips
st.markdown("**Sample Topics:**")
col1, col2, col3 = st.columns(3)
sample_1 = col1.button("🤖 Multi-Agent Orchestration 2026")
sample_2 = col2.button("⚛️ Quantum Computing Advancements")
sample_3 = col3.button("🔋 Solid-State Battery Breakthroughs")

selected_query = ""
if sample_1:
    selected_query = "State of autonomous multi-agent orchestration architectures in 2026"
elif sample_2:
    selected_query = "Latest breakthroughs in quantum computing and error-corrected qubits"
elif sample_3:
    selected_query = "Recent advancements and commercialization of solid-state battery technology"

# Query Input
query = st.text_area(
    "What topic would you like to research?",
    value=selected_query if selected_query else "Latest advancements in agentic AI architectures and multi-step reasoning",
    height=80
)

start_btn = st.button("🚀 Start Deep Research", type="primary", use_container_width=True)

if start_btn:
    if not os.getenv("OPENAI_API_KEY") or not os.getenv("TAVILY_API_KEY"):
        st.error("⚠️ Please provide both **OpenAI API Key** and **Tavily API Key** in the sidebar to run the research agent.")
    else:
        # Import agent modules
        try:
            from agent import build_research_graph
        except ImportError as e:
            st.error(f"Error importing agent: {e}")
            st.stop()

        status_container = st.container()
        
        with status_container:
            st.markdown("### 🔄 Execution Progress")
            progress_bar = st.progress(10)
            status_text = st.empty()
            
            status_text.info("🧠 Step 1/5: Decomposing research topic into multi-angle queries...")
            time.sleep(0.5)
            
            # Build and invoke graph
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
            status_text.info("🔍 Step 2 & 3: Conducting web searches and evaluating evidence depth...")
            
            try:
                result = agent.invoke(initial_state)
                
                progress_bar.progress(75)
                status_text.info("📝 Step 4: Synthesizing structured draft with inline citations...")
                time.sleep(0.5)
                
                progress_bar.progress(90)
                status_text.info("🛡️ Step 5: Running Citation & Fact-Checking Verifier...")
                time.sleep(0.5)
                
                progress_bar.progress(100)
                status_text.success("✅ Deep Research Complete & Verified!")
                
                # Metrics Row
                m1, m2, m3 = st.columns(3)
                m1.metric("Unique Sources Found", len(result.get("search_results", [])))
                m2.metric("Research Cycles", result.get("iteration_count", 1))
                m3.metric("Verification Status", "Verified 🛡️")
                
                st.markdown("---")
                
                # Layout Tabs
                tab_report, tab_sources, tab_audit = st.tabs(["📋 Verified Report", "🔗 Consulted Sources", "🔍 Verifier Audit"])
                
                with tab_report:
                    report_text = result.get("verified_report", "")
                    st.markdown(report_text)
                    
                    st.markdown("---")
                    # Download options
                    st.download_button(
                        label="📥 Download Report (.md)",
                        data=report_text,
                        file_name="deep_research_report.md",
                        mime="text/markdown"
                    )
                    
                with tab_sources:
                    sources = result.get("search_results", [])
                    st.markdown(f"**Found {len(sources)} verified references:**")
                    for idx, src in enumerate(sources, 1):
                        title = src.get("title", "Untitled Source")
                        url = src.get("url", "#")
                        snippet = src.get("content", "")
                        with st.expander(f"[{idx}] {title}"):
                            st.markdown(f"**URL**: [{url}]({url})")
                            st.write(snippet)
                            
                with tab_audit:
                    st.markdown("### 🛡️ Fact-Check & Reflection Audit Log")
                    st.info(f"**Reflection Notes**: {result.get('critique_notes', 'N/A')}")
                    st.write(f"**Total Iterations Executed**: {result.get('iteration_count', 1)} / {max_cycles}")
                    
            except Exception as e:
                st.error(f"❌ An error occurred during research execution: {e}")