"""
Streamlit Web Interface for Autonomous Deep Research Agent.

Features:
- Live graph execution streaming via LangGraph stream updates
- Real-time display of query decomposition, source gathering, and reflection
- Deterministic citation verification and audit inspection
"""

import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="Deep Research Agent",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Sidebar Configuration
with st.sidebar:
    st.title("Configuration")
    
    env_openai = os.getenv("OPENAI_API_KEY", "")
    env_tavily = os.getenv("TAVILY_API_KEY", "")
    
    openai_key = st.text_input(
        "OpenAI API Key",
        value=env_openai,
        type="password",
        help="Used for decomposition, synthesis, and verification",
    )
    tavily_key = st.text_input(
        "Tavily API Key",
        value=env_tavily,
        type="password",
        help="Used for real-time web search retrieval",
    )
    
    if openai_key:
        os.environ["OPENAI_API_KEY"] = openai_key
    if tavily_key:
        os.environ["TAVILY_API_KEY"] = tavily_key
        
    st.divider()
    model_choice = st.selectbox(
        "Model",
        ["gpt-4o-mini", "gpt-4o"],
        index=0,
    )
    os.environ["OPENAI_MODEL"] = model_choice
    
    max_cycles = st.slider(
        "Max Reflection Cycles",
        min_value=1,
        max_value=3,
        value=2,
        help="Maximum times the agent evaluates evidence and searches for missing information",
    )

# Header
st.title("Autonomous Deep Research Agent")
st.caption("Cyclic LangGraph state machine with search decomposition, self-reflection, and citation verification.")

# Sample queries
st.markdown("**Quick Topics:**")
col1, col2, col3 = st.columns(3)
selected_topic = ""
if col1.button("Multi-Agent Orchestration Patterns"):
    selected_topic = "State of multi-agent orchestration architectures and communication protocols in 2026"
elif col2.button("Quantum Computing & Error Correction"):
    selected_topic = "Recent breakthroughs in quantum computing and logical qubit error mitigation"
elif col3.button("Solid-State Battery Progress"):
    selected_topic = "Current commercialization milestones and technical hurdles for solid-state EV batteries"

# Query Input
query = st.text_area(
    "Research Topic",
    value=selected_topic if selected_topic else "Advances in LLM reasoning techniques and test-time compute scaling",
    height=80,
)

execute_clicked = st.button("Start Deep Research", type="primary", use_container_width=True)

if execute_clicked:
    if not os.getenv("OPENAI_API_KEY") or not os.getenv("TAVILY_API_KEY"):
        st.error("Please configure both OPENAI_API_KEY and TAVILY_API_KEY in the sidebar.")
    else:
        try:
            from agent import build_research_graph
        except ImportError as exc:
            st.error(f"Failed to import research graph: {exc}")
            st.stop()

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
            "citation_audit": {},
        }

        # Track accumulated state across streaming updates
        all_sub_queries = []
        all_sources = []
        reflection_notes = []
        final_report = ""
        final_audit = {}

        # Native LangGraph event stream
        with st.status("Executing Research Workflow...", expanded=True) as status:
            try:
                for update in agent.stream(initial_state, stream_mode="updates"):
                    for node_name, state_delta in update.items():
                        if node_name == "decompose":
                            sub_q = state_delta.get("sub_queries", [])
                            all_sub_queries.extend(sub_q)
                            st.write(f"**Step 1: Decomposed into {len(sub_q)} sub-queries:**")
                            for q in sub_q:
                                st.markdown(f"- *{q}*")

                        elif node_name == "search":
                            all_sources = state_delta.get("search_results", [])
                            st.write(f"**Step 2: Search executed** — {len(all_sources)} unique sources aggregated.")

                        elif node_name == "evaluate":
                            notes = state_delta.get("critique_notes", "")
                            is_sufficient = state_delta.get("is_sufficient", True)
                            reflection_notes.append(notes)
                            if is_sufficient:
                                st.write(f"**Step 3: Reflection** — Evidence sufficient. Proceeding to synthesis.")
                            else:
                                new_sub_q = state_delta.get("sub_queries", [])
                                st.write(f"**Step 3: Reflection** — Missing gaps identified. Initiating follow-up search:")
                                for q in new_sub_q:
                                    st.markdown(f"- *{q}*")

                        elif node_name == "synthesize":
                            st.write("**Step 4: Draft report synthesized with inline citations.**")

                        elif node_name == "verify":
                            final_report = state_delta.get("verified_report", "")
                            final_audit = state_delta.get("citation_audit", {})
                            st.write("**Step 5: Citation audit complete.** All claims cross-referenced.")

                status.update(label="Research Complete!", state="complete", expanded=False)
            except Exception as e:
                status.update(label="Execution Error", state="error")
                st.error(f"Error during agent execution: {e}")
                st.stop()

        # Display Metrics
        m1, m2, m3 = st.columns(3)
        m1.metric("Unique Sources", len(all_sources))
        m2.metric("Sub-Queries Evaluated", len(all_sub_queries))
        m3.metric("Invalid Citations Detected", len(final_audit.get("invalid_indices", [])))

        st.divider()

        # Render Results
        tab_report, tab_sources, tab_audit = st.tabs(["Research Report", "Sources", "Verification Audit"])

        with tab_report:
            if final_report:
                st.markdown(final_report)
                st.download_button(
                    label="Download Report (Markdown)",
                    data=final_report,
                    file_name="research_report.md",
                    mime="text/markdown",
                )
            else:
                st.info("No report generated.")

        with tab_sources:
            st.markdown(f"### Gathered Sources ({len(all_sources)})")
            for idx, src in enumerate(all_sources, 1):
                title = src.get("title", f"Source [{idx}]")
                url = src.get("url", "#")
                snippet = src.get("content", "No excerpt available.")
                with st.expander(f"[{idx}] {title}"):
                    st.markdown(f"**URL:** [{url}]({url})")
                    st.write(snippet)

        with tab_audit:
            st.markdown("### Citation & Grounding Analysis")
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**Cited References in Draft:**")
                st.write(final_audit.get("cited_indices", []))
            with col_b:
                st.markdown("**Out-of-Bounds Citations (Flagged):**")
                invalid = final_audit.get("invalid_indices", [])
                if invalid:
                    st.warning(f"Detected invalid source references: {invalid}")
                else:
                    st.success("All cited references mapped to retrieved sources.")

            st.markdown("**Reflection Notes:**")
            for idx, note in enumerate(reflection_notes, 1):
                st.info(f"**Cycle {idx}:** {note}")