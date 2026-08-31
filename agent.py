"""
Autonomous Deep Research Agent with Iterative Refinement & Citation Verification.

Framework: LangGraph + LangChain + Tavily Search
Author: Vikas Joshi <vikasjoshi.2027@gmail.com>

Architecture:
1. Query Decomposition: Breaks broad research topics into targeted sub-queries.
2. Multi-Source Search: Gathers web intelligence and deduplicates results.
3. Reflection & Gap Analysis: Evaluates evidence sufficiency and triggers follow-up searches if needed.
4. Structured Synthesis: Drafts an executive research report with inline citations.
5. Citation & Fact Verification: Audits every claim against retrieved evidence to eliminate hallucinations.
"""

import argparse
import json
import os
import sys
from typing import Annotated, Any, Dict, List, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

# Load environment configuration
load_dotenv()


class ResearchState(TypedDict):
    """Represents the complete state of the autonomous deep research workflow."""
    messages: Annotated[list, add_messages]
    query: str
    sub_queries: List[str]
    search_results: List[Dict[str, Any]]
    iteration_count: int
    max_iterations: int
    is_sufficient: bool
    critique_notes: str
    draft_report: str
    verified_report: str
    verification_audit: str


def get_llm(temperature: float = 0.2) -> ChatOpenAI:
    """Helper to instantiate the primary LLM model."""
    model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    return ChatOpenAI(model=model_name, temperature=temperature)


def decompose_query(state: ResearchState) -> Dict[str, Any]:
    """Decomposes the primary research topic into targeted sub-queries."""
    query = state["query"]
    print(f"\n🧠 [1/5] Decomposing research topic: '{query}'...")

    prompt = f"""You are a senior research strategist. Analyze the user's research topic and generate 3 distinct, highly targeted search queries to thoroughly investigate different angles (e.g. background/state-of-the-art, key technical breakthroughs/applications, and current challenges/future outlook).

Research Topic: {query}

Return ONLY a JSON array of 3 search queries, with no extra text or markdown formatting. Example:
["query 1", "query 2", "query 3"]
"""

    llm = get_llm(temperature=0.3)
    response = llm.invoke([
        SystemMessage(content="You generate structured JSON search queries."),
        HumanMessage(content=prompt)
    ])

    content = response.content.strip()
    # Remove markdown code block fences if present
    if content.startswith("```"):
        lines = content.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        content = "\n".join(lines).strip()

    try:
        sub_queries = json.loads(content)
        if not isinstance(sub_queries, list) or len(sub_queries) == 0:
            sub_queries = [query]
    except Exception:
        sub_queries = [f"{query} overview", f"{query} breakthroughs", f"{query} challenges"]

    print("   Generated Sub-Queries:")
    for idx, q in enumerate(sub_queries, 1):
        print(f"   {idx}. {q}")

    return {
        "sub_queries": sub_queries,
        "iteration_count": state.get("iteration_count", 0) + 1
    }


def execute_searches(state: ResearchState) -> Dict[str, Any]:
    """Executes search queries on Tavily, deduplicating findings by URL."""
    sub_queries = state.get("sub_queries", [state["query"]])
    existing_results = state.get("search_results", [])
    seen_urls = {r.get("url") for r in existing_results if r.get("url")}
    new_results = list(existing_results)

    print(f"\n🔍 [2/5] Executing web searches for {len(sub_queries)} queries (Iteration {state.get('iteration_count', 1)})...")

    try:
        tool = TavilySearch(max_results=3)
        for q in sub_queries:
            print(f"   -> Searching: '{q}'")
            raw = tool.invoke(q)
            items = []
            if isinstance(raw, dict):
                items = raw.get("results", [])
            elif isinstance(raw, list):
                items = raw

            for item in items:
                url = item.get("url")
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    new_results.append(item)
    except Exception as exc:
        print(f"   ⚠️ Search API warning: {exc}")

    print(f"   ✅ Accumulated {len(new_results)} unique source references.")
    return {"search_results": new_results}


def evaluate_completeness(state: ResearchState) -> Dict[str, Any]:
    """Critique & Reflection Node: Evaluates if sufficient evidence was gathered or if gaps remain."""
    query = state["query"]
    results = state.get("search_results", [])
    current_iteration = state.get("iteration_count", 1)
    max_iterations = state.get("max_iterations", 2)

    print(f"\n🧐 [3/5] Evaluating evidence completeness (Cycle {current_iteration}/{max_iterations})...")

    # If reached max depth, mark as sufficient
    if current_iteration >= max_iterations:
        print("   Reached maximum research iterations. Proceeding to report synthesis.")
        return {"is_sufficient": True, "critique_notes": "Max iterations reached."}

    # Summary of available evidence
    snippets = [f"- {r.get('title')}: {r.get('content', '')[:200]}..." for r in results[:8]]
    context = "\n".join(snippets)

    prompt = f"""You are a rigorous research auditor.
Target Topic: {query}
Retrieved Sources Summary:
{context}

Analyze whether the retrieved evidence is sufficiently deep, accurate, and comprehensive to write a high-grade research report.
Respond in valid JSON format only:
{{
  "is_sufficient": true/false,
  "reasoning": "Brief explanation of gaps or coverage",
  "follow_up_queries": ["query 1", "query 2"] (if is_sufficient is false, provide 1-2 specific follow-up queries to fill gaps; otherwise empty list)
}}
"""

    llm = get_llm(temperature=0.1)
    response = llm.invoke([
        SystemMessage(content="You evaluate research data completeness and respond strictly in JSON."),
        HumanMessage(content=prompt)
    ])

    content = response.content.strip()
    if content.startswith("```"):
        lines = content.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        content = "\n".join(lines).strip()

    try:
        evaluation = json.loads(content)
        is_sufficient = evaluation.get("is_sufficient", True)
        follow_up = evaluation.get("follow_up_queries", [])
        critique = evaluation.get("reasoning", "")
    except Exception:
        is_sufficient = True
        follow_up = []
        critique = "Defaulted to complete."

    if not is_sufficient and follow_up:
        print(f"   ⚠️ Information gaps detected: {critique}")
        print(f"   🔄 Scheduling follow-up search: {follow_up}")
        return {
            "is_sufficient": False,
            "sub_queries": follow_up,
            "critique_notes": critique,
            "iteration_count": current_iteration + 1
        }
    else:
        print(f"   ✅ Evidence evaluated as comprehensive. {critique}")
        return {
            "is_sufficient": True,
            "critique_notes": critique
        }


def synthesize_draft(state: ResearchState) -> Dict[str, Any]:
    """Synthesizes all gathered evidence into a structured draft report with inline citations."""
    query = state["query"]
    results = state.get("search_results", [])
    print("\n📝 [4/5] Synthesizing comprehensive draft report with inline citations...")

    context_lines = []
    for idx, r in enumerate(results, 1):
        title = r.get("title", "Untitled Source")
        url = r.get("url", "N/A")
        content = r.get("content", "")
        context_lines.append(f"Source [{idx}]: {title}\nURL: {url}\nContent: {content}\n")

    context = "\n".join(context_lines) if context_lines else "No search evidence available."

    prompt = f"""You are a Lead AI Research Analyst.
Write an authoritative, highly comprehensive research report on the following topic based solely on the provided numbered sources.

Topic: {query}

Evidence Sources:
{context}

Requirements:
1. Use inline citations [1], [2], etc., for EVERY factual claim, statistic, or architectural insight.
2. Structure the report with:
   # Executive Summary
   # Core Breakthroughs & Technological Foundations
   # Detailed Analysis & Comparative Insights
   # Real-World Applications & Case Studies
   # Challenges, Trade-offs & Future Outlook
   # Bibliography & Cited Sources
3. Maintain an academic, professional, and data-driven tone.
"""

    llm = get_llm(temperature=0.2)
    response = llm.invoke([
        SystemMessage(content="You are an expert technical researcher producing cited reports."),
        HumanMessage(content=prompt)
    ])

    return {"draft_report": response.content}


def verify_citations(state: ResearchState) -> Dict[str, Any]:
    """Fact-Checking & Hallucination Verifier Node: Cross-checks claims against source texts."""
    draft = state.get("draft_report", "")
    results = state.get("search_results", [])
    print("\n🛡️ [5/5] Running Hallucination Verifier & Citation Audit...")

    source_map = []
    for idx, r in enumerate(results, 1):
        source_map.append(f"[{idx}] {r.get('title')}: {r.get('content', '')[:300]}")
    sources_summary = "\n".join(source_map)

    prompt = f"""You are a Citation Verification and Fact-Checking Auditor.
Review the following draft report and verify that:
1. Every claim with an inline citation [X] is supported by the corresponding source summary.
2. Any unverified or hallucinated claims are corrected or removed.
3. The bibliography matches the citations accurately.

Source Summaries:
{sources_summary}

Draft Report:
{draft}

Produce the final verified report followed by a brief '### 🔍 Verification & Fact-Check Audit' section detailing any corrections made and confidence rating (High / Medium).
"""

    llm = get_llm(temperature=0.1)
    response = llm.invoke([
        SystemMessage(content="You are an uncompromising fact-checker and citation verifier."),
        HumanMessage(content=prompt)
    ])

    verified_content = response.content

    return {
        "verified_report": verified_content,
        "messages": [response]
    }


def route_research(state: ResearchState) -> str:
    """Conditional routing edge: Determines whether to loop for more searches or proceed to draft."""
    if state.get("is_sufficient", True):
        return "synthesize"
    return "search"


def build_research_graph():
    """Builds and compiles the cyclic LangGraph workflow with reflection and verification."""
    workflow = StateGraph(ResearchState)

    # Add Nodes
    workflow.add_node("decompose", decompose_query)
    workflow.add_node("search", execute_searches)
    workflow.add_node("evaluate", evaluate_completeness)
    workflow.add_node("synthesize", synthesize_draft)
    workflow.add_node("verify", verify_citations)

    # Define Graph Edges
    workflow.set_entry_point("decompose")
    workflow.add_edge("decompose", "search")
    workflow.add_edge("search", "evaluate")

    # Conditional Branch: Loop back to search if gaps exist, or synthesize if sufficient
    workflow.add_conditional_edges(
        "evaluate",
        route_research,
        {
            "search": "search",
            "synthesize": "synthesize"
        }
    )

    workflow.add_edge("synthesize", "verify")
    workflow.add_edge("verify", END)

    return workflow.compile()


def main():
    parser = argparse.ArgumentParser(
        description="Autonomous Deep Research Agent with Iterative Refinement & Citation Verification"
    )
    parser.add_argument(
        "--query",
        type=str,
        default="Architectures and reasoning patterns in autonomous multi-agent AI systems (2025/2026)",
        help="Research topic or question to investigate",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=2,
        help="Maximum search & reflection cycles (default: 2)",
    )
    args = parser.parse_args()

    # Verify environment keys
    openai_key = os.getenv("OPENAI_API_KEY")
    tavily_key = os.getenv("TAVILY_API_KEY")

    if not openai_key or not tavily_key:
        print("⚠️ Warning: Missing API keys in environment.")
        if not openai_key:
            print("   - OPENAI_API_KEY is not set")
        if not tavily_key:
            print("   - TAVILY_API_KEY is not set")
        print("   Please create a .env file with your keys before running live searches.\n")

    agent = build_research_graph()

    initial_state = {
        "messages": [],
        "query": args.query,
        "sub_queries": [],
        "search_results": [],
        "iteration_count": 0,
        "max_iterations": args.max_iterations,
        "is_sufficient": False,
        "critique_notes": "",
        "draft_report": "",
        "verified_report": "",
        "verification_audit": ""
    }

    print("=" * 75)
    print("🚀 STARTING AUTONOMOUS DEEP RESEARCH AGENT")
    print(f"📌 Research Topic: {args.query}")
    print(f"⚙️ Max Iterations: {args.max_iterations}")
    print("=" * 75)

    output = agent.invoke(initial_state)

    print("\n" + "=" * 75)
    print("📋 FINAL VERIFIED RESEARCH REPORT & AUDIT")
    print("=" * 75 + "\n")
    print(output["verified_report"])


if __name__ == "__main__":
    main()