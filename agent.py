"""
Autonomous Deep Research Agent with Iterative Refinement and Citation Verification.

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
    """Helper function to instantiate the primary LLM model."""
    model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    return ChatOpenAI(model=model_name, temperature=temperature)


def decompose_query(state: ResearchState) -> Dict[str, Any]:
    """Decomposes the primary research topic into targeted sub-queries."""
    query = state["query"]
    print(f"\n[INFO] [Step 1/5] Decomposing research topic: '{query}'...")

    prompt = f"""You are a senior technical research strategist. Analyze the user's research topic and generate 3 distinct, targeted search queries covering:
1. Background and foundational concepts
2. Recent technical breakthroughs and state-of-the-art developments
3. Current limitations, challenges, and future outlook

Research Topic: {query}

Return ONLY a JSON array of 3 strings. Example:
["query 1", "query 2", "query 3"]
"""

    llm = get_llm(temperature=0.3)
    response = llm.invoke([
        SystemMessage(content="You generate structured JSON search queries."),
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
        sub_queries = json.loads(content)
        if not isinstance(sub_queries, list) or len(sub_queries) == 0:
            sub_queries = [query]
    except Exception:
        sub_queries = [f"{query} overview", f"{query} breakthroughs", f"{query} challenges"]

    print("       Generated sub-queries:")
    for idx, q in enumerate(sub_queries, 1):
        print(f"       {idx}. {q}")

    return {
        "sub_queries": sub_queries,
        "iteration_count": state.get("iteration_count", 0) + 1
    }


def execute_searches(state: ResearchState) -> Dict[str, Any]:
    """Executes search queries using Tavily, deduplicating findings by URL."""
    sub_queries = state.get("sub_queries", [state["query"]])
    existing_results = state.get("search_results", [])
    seen_urls = {r.get("url") for r in existing_results if r.get("url")}
    new_results = list(existing_results)

    current_iter = state.get("iteration_count", 1)
    print(f"\n[INFO] [Step 2/5] Executing search across {len(sub_queries)} queries (Iteration {current_iter})...")

    try:
        tool = TavilySearch(max_results=3)
        for q in sub_queries:
            print(f"       - Querying: '{q}'")
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
        print(f"[WARNING] Search API encountered an exception: {exc}")

    print(f"       Total unique sources aggregated: {len(new_results)}")
    return {"search_results": new_results}


def evaluate_completeness(state: ResearchState) -> Dict[str, Any]:
    """Critique and Reflection: Evaluates if sufficient evidence was gathered or if gaps remain."""
    query = state["query"]
    results = state.get("search_results", [])
    current_iteration = state.get("iteration_count", 1)
    max_iterations = state.get("max_iterations", 2)

    print(f"\n[INFO] [Step 3/5] Evaluating evidence completeness (Cycle {current_iteration}/{max_iterations})...")

    if current_iteration >= max_iterations:
        print("       Maximum iteration depth reached. Proceeding to synthesis.")
        return {"is_sufficient": True, "critique_notes": "Maximum iterations reached."}

    snippets = [f"- {r.get('title')}: {r.get('content', '')[:200]}..." for r in results[:8]]
    context = "\n".join(snippets)

    prompt = f"""You are a research auditor evaluating information sufficiency.
Target Topic: {query}

Retrieved Evidence Summaries:
{context}

Determine whether the gathered evidence is comprehensive and deep enough to draft a thorough, authoritative technical report.
Respond strictly in valid JSON:
{{
  "is_sufficient": true/false,
  "reasoning": "Explanation of coverage or missing gaps",
  "follow_up_queries": ["query 1", "query 2"]
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
        critique = "Evaluation defaulted to complete."

    if not is_sufficient and follow_up:
        print(f"       Information gaps identified: {critique}")
        print(f"       Scheduling follow-up queries: {follow_up}")
        return {
            "is_sufficient": False,
            "sub_queries": follow_up,
            "critique_notes": critique,
            "iteration_count": current_iteration + 1
        }
    else:
        print(f"       Evidence evaluated as sufficient. {critique}")
        return {
            "is_sufficient": True,
            "critique_notes": critique
        }


def synthesize_draft(state: ResearchState) -> Dict[str, Any]:
    """Synthesizes gathered evidence into a structured draft report with inline citations."""
    query = state["query"]
    results = state.get("search_results", [])
    print("\n[INFO] [Step 4/5] Synthesizing comprehensive research report draft with citations...")

    context_lines = []
    for idx, r in enumerate(results, 1):
        title = r.get("title", "Untitled Source")
        url = r.get("url", "N/A")
        content = r.get("content", "")
        context_lines.append(f"Source [{idx}]: {title}\nURL: {url}\nContent: {content}\n")

    context = "\n".join(context_lines) if context_lines else "No search evidence available."

    prompt = f"""You are a senior technical research analyst.
Write an authoritative, rigorous, and well-structured technical research report on the topic below based on the provided numbered sources.

Topic: {query}

Evidence Sources:
{context}

Requirements:
1. Include inline citations [1], [2], etc., for every factual statement, statistic, or architectural finding.
2. Structure the report as follows:
   # Executive Summary
   # Core Concepts and Architectural Foundations
   # Detailed Findings and Technical Analysis
   # Real-World Applications and Case Studies
   # Challenges, Trade-offs, and Future Outlook
   # References and Cited Sources
3. Maintain an academic, precise, and professional tone throughout.
"""

    llm = get_llm(temperature=0.2)
    response = llm.invoke([
        SystemMessage(content="You are an expert technical researcher producing cited reports."),
        HumanMessage(content=prompt)
    ])

    return {"draft_report": response.content}


def verify_citations(state: ResearchState) -> Dict[str, Any]:
    """Fact-Checking and Citation Verification: Cross-checks claims against source texts."""
    draft = state.get("draft_report", "")
    results = state.get("search_results", [])
    print("\n[INFO] [Step 5/5] Auditing citations and factual consistency against source text...")

    source_map = []
    for idx, r in enumerate(results, 1):
        source_map.append(f"[{idx}] {r.get('title')}: {r.get('content', '')[:300]}")
    sources_summary = "\n".join(source_map)

    prompt = f"""You are a Citation Verification and Fact-Checking Auditor.
Review the following draft report and verify that:
1. Every claim with an inline citation [X] is supported by the corresponding source summary.
2. Any unverified or hallucinated claims are corrected or removed.
3. The bibliography accurately matches the cited references.

Source Summaries:
{sources_summary}

Draft Report:
{draft}

Produce the final verified report followed by a brief '### Verification Audit' section detailing any corrections made and a confidence score (High / Medium).
"""

    llm = get_llm(temperature=0.1)
    response = llm.invoke([
        SystemMessage(content="You are a meticulous citation auditor and fact-checker."),
        HumanMessage(content=prompt)
    ])

    return {
        "verified_report": response.content,
        "messages": [response]
    }


def route_research(state: ResearchState) -> str:
    """Conditional routing edge: Determines whether to execute additional searches or proceed to draft."""
    if state.get("is_sufficient", True):
        return "synthesize"
    return "search"


def build_research_graph():
    """Builds and compiles the cyclic LangGraph workflow."""
    workflow = StateGraph(ResearchState)

    # Register Nodes
    workflow.add_node("decompose", decompose_query)
    workflow.add_node("search", execute_searches)
    workflow.add_node("evaluate", evaluate_completeness)
    workflow.add_node("synthesize", synthesize_draft)
    workflow.add_node("verify", verify_citations)

    # Register Edges
    workflow.set_entry_point("decompose")
    workflow.add_edge("decompose", "search")
    workflow.add_edge("search", "evaluate")

    # Conditional Branching
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
        description="Autonomous Deep Research Agent with Iterative Refinement and Citation Verification"
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
        help="Maximum search and reflection cycles (default: 2)",
    )
    args = parser.parse_args()

    # Validate environment credentials
    openai_key = os.getenv("OPENAI_API_KEY")
    tavily_key = os.getenv("TAVILY_API_KEY")

    if not openai_key or not tavily_key:
        print("[WARNING] Missing API credentials in environment.")
        if not openai_key:
            print("          - OPENAI_API_KEY is not set")
        if not tavily_key:
            print("          - TAVILY_API_KEY is not set")
        print("          Ensure .env is configured before running live searches.\n")

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
    print("Autonomous Deep Research Agent")
    print(f"Topic: {args.query}")
    print(f"Max Iterations: {args.max_iterations}")
    print("=" * 75)

    output = agent.invoke(initial_state)

    print("\n" + "=" * 75)
    print("Verified Research Report and Audit")
    print("=" * 75 + "\n")
    print(output["verified_report"])


if __name__ == "__main__":
    main()