"""
Autonomous Deep Research Agent with Iterative Refinement and Citation Verification.

Architecture:
1. Decompose: Splits a broad research prompt into focused sub-queries.
2. Search: Executes web search via Tavily and deduplicates results by URL.
3. Evaluate: Identifies information gaps and determines whether another search loop is needed.
4. Synthesize: Compiles findings into a structured technical draft with inline citations.
5. Verify: Programmatically checks citation bounds and audits factual consistency against source passages.
"""

import argparse
import json
import os
import re
import sys
from typing import Annotated, Any, Dict, List, Optional, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

load_dotenv()


class ResearchState(TypedDict):
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
    citation_audit: Dict[str, Any]


def get_llm(temperature: float = 0.2) -> ChatOpenAI:
    """Instantiate ChatOpenAI using environment configuration."""
    model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    return ChatOpenAI(model=model_name, temperature=temperature)


def parse_json_from_response(text: str) -> Any:
    """Extracts and parses JSON from model output, handling markdown code fences."""
    cleaned = text.strip()
    if "```" in cleaned:
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned, re.IGNORECASE)
        if match:
            cleaned = match.group(1).strip()
    return json.loads(cleaned)


def extract_citation_indices(text: str) -> List[int]:
    """Extracts unique citation indices like [1], [2] from text."""
    matches = re.findall(r"\[(\d+)\]", text)
    return sorted(list(set(int(m) for m in matches)))


def decompose_query(state: ResearchState) -> Dict[str, Any]:
    """Breaks down the main research query into 3 distinct investigative angles."""
    query = state["query"]
    prompt = f"""Break down this research topic into 3 specific, non-overlapping search queries to investigate:
1. Core mechanisms, foundations, and current standard practice
2. Recent developments, breakthroughs, or state-of-the-art updates
3. Key technical limitations, trade-offs, and open challenges

Topic: {query}

Respond ONLY with a valid JSON array of 3 strings. Example:
["query 1", "query 2", "query 3"]
"""

    llm = get_llm(temperature=0.2)
    response = llm.invoke([
        SystemMessage(content="You generate structured JSON search queries. Output valid JSON only."),
        HumanMessage(content=prompt),
    ])

    try:
        sub_queries = parse_json_from_response(response.content)
        if not isinstance(sub_queries, list) or len(sub_queries) == 0:
            raise ValueError("Expected a non-empty list")
    except Exception:
        # Fallback to sensible defaults if JSON parsing fails
        sub_queries = [
            f"{query} architecture and fundamentals",
            f"{query} recent developments and benchmarks",
            f"{query} limitations and trade-offs",
        ]

    return {
        "sub_queries": sub_queries,
        "iteration_count": state.get("iteration_count", 0) + 1,
    }


def execute_searches(state: ResearchState) -> Dict[str, Any]:
    """Queries Tavily and appends deduplicated results based on URL."""
    sub_queries = state.get("sub_queries", [state["query"]])
    existing_results = state.get("search_results", [])
    seen_urls = {r.get("url") for r in existing_results if r.get("url")}
    new_results = list(existing_results)

    try:
        tavily = TavilySearch(max_results=3)
        for q in sub_queries:
            raw = tavily.invoke(q)
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
        print(f"[Search Warning] Query execution failed: {exc}", file=sys.stderr)

    return {"search_results": new_results}


def evaluate_completeness(state: ResearchState) -> Dict[str, Any]:
    """Reflection step: Audits gathered evidence to detect gaps and determine termination."""
    query = state["query"]
    results = state.get("search_results", [])
    current_iteration = state.get("iteration_count", 1)
    max_iterations = state.get("max_iterations", 2)

    # Deterministic recursion guard
    if current_iteration >= max_iterations:
        return {
            "is_sufficient": True,
            "critique_notes": f"Reached maximum iteration limit ({max_iterations}). Proceeding to synthesis.",
        }

    evidence_excerpts = []
    for idx, r in enumerate(results[:10], 1):
        title = r.get("title", "Untitled")
        content = r.get("content", "")[:350].strip()
        evidence_excerpts.append(f"[{idx}] {title}: {content}")
    context = "\n".join(evidence_excerpts)

    prompt = f"""Evaluate whether the gathered evidence provides sufficient technical depth to draft a comprehensive report on:
Topic: {query}

Retrieved Evidence Summaries:
{context}

Respond strictly in JSON:
{{
  "is_sufficient": boolean,
  "reasoning": "brief explanation of what is covered vs missing",
  "follow_up_queries": ["query 1", "query 2"]
}}
If is_sufficient is true, leave follow_up_queries empty.
"""

    llm = get_llm(temperature=0.1)
    response = llm.invoke([
        SystemMessage(content="You evaluate research data completeness and respond strictly in JSON."),
        HumanMessage(content=prompt),
    ])

    try:
        data = parse_json_from_response(response.content)
        is_sufficient = data.get("is_sufficient", True)
        follow_up = data.get("follow_up_queries", [])
        critique = data.get("reasoning", "Evaluation complete.")
    except Exception:
        is_sufficient = True
        follow_up = []
        critique = "Defaulted to sufficient after parse error."

    if not is_sufficient and follow_up:
        return {
            "is_sufficient": False,
            "sub_queries": follow_up[:3],
            "critique_notes": critique,
            "iteration_count": current_iteration + 1,
        }

    return {
        "is_sufficient": True,
        "critique_notes": critique,
    }


def synthesize_draft(state: ResearchState) -> Dict[str, Any]:
    """Compiles gathered sources into a structured technical draft with inline citations."""
    query = state["query"]
    results = state.get("search_results", [])

    source_blocks = []
    for idx, r in enumerate(results, 1):
        title = r.get("title", "Untitled")
        url = r.get("url", "N/A")
        content = r.get("content", "").strip()
        source_blocks.append(f"Source [{idx}]: {title}\nURL: {url}\nExcerpt: {content}\n")

    context = "\n---\n".join(source_blocks) if source_blocks else "No search evidence available."

    prompt = f"""Write an in-depth technical research report on: {query}

Use ONLY the provided evidence sources below. You MUST cite sources using inline brackets like [1], [2] when referencing factual claims, data, or design choices.

Evidence Sources:
{context}

Structure:
# Executive Summary
# Architectural Overview & Core Concepts
# Technical Findings & Analysis
# Real-World Applications & Implementations
# Open Challenges & Future Outlook
# References

Ensure every cited number [X] corresponds to a valid source listed above.
"""

    llm = get_llm(temperature=0.2)
    response = llm.invoke([
        SystemMessage(content="You are a technical research writer. You write precise reports with grounded citations."),
        HumanMessage(content=prompt),
    ])

    return {"draft_report": response.content}


def verify_citations(state: ResearchState) -> Dict[str, Any]:
    """Performs deterministic citation checking and verifies factual alignment with sources."""
    draft = state.get("draft_report", "")
    results = state.get("search_results", [])

    # 1. Deterministic citation bounds checking
    cited_indices = extract_citation_indices(draft)
    valid_range = set(range(1, len(results) + 1))
    invalid_indices = [idx for idx in cited_indices if idx not in valid_range]

    # 2. Build full context for sources cited in the draft
    source_context_list = []
    for idx, r in enumerate(results, 1):
        # Include substantial context (up to 1200 chars) for thorough cross-checking
        snippet = r.get("content", "")[:1200].strip()
        source_context_list.append(f"Source [{idx}] ({r.get('title', 'N/A')}):\n{snippet}\n")
    source_context = "\n".join(source_context_list)

    prompt = f"""Review the following draft report and audit every cited claim against the provided source passages.

Source Passages:
{source_context}

Draft Report:
{draft}

Your Tasks:
1. Verify that claims attributed to [X] are accurately reflected in Source [X].
2. If any claim is misattributed or unsupported, refine or qualify the text.
3. Keep valid inline citations intact.
4. At the very end of the report, append an '### Citation & Fact-Check Audit' section with:
   - Total sources cited vs available
   - Any claims refined during verification
   - Verification Confidence: High or Medium
"""

    llm = get_llm(temperature=0.1)
    response = llm.invoke([
        SystemMessage(content="You are a factual verification auditor. You preserve truth and accurate citations."),
        HumanMessage(content=prompt),
    ])

    audit_summary = {
        "cited_indices": cited_indices,
        "invalid_indices": invalid_indices,
        "total_sources_available": len(results),
        "out_of_bounds_detected": len(invalid_indices) > 0,
    }

    return {
        "verified_report": response.content,
        "citation_audit": audit_summary,
        "messages": [response],
    }


def route_research(state: ResearchState) -> str:
    """Routes back to search if gaps remain, otherwise proceeds to synthesis."""
    if state.get("is_sufficient", True):
        return "synthesize"
    return "search"


def build_research_graph():
    """Builds and compiles the cyclic LangGraph workflow."""
    workflow = StateGraph(ResearchState)

    workflow.add_node("decompose", decompose_query)
    workflow.add_node("search", execute_searches)
    workflow.add_node("evaluate", evaluate_completeness)
    workflow.add_node("synthesize", synthesize_draft)
    workflow.add_node("verify", verify_citations)

    workflow.set_entry_point("decompose")
    workflow.add_edge("decompose", "search")
    workflow.add_edge("search", "evaluate")

    workflow.add_conditional_edges(
        "evaluate",
        route_research,
        {
            "search": "search",
            "synthesize": "synthesize",
        },
    )

    workflow.add_edge("synthesize", "verify")
    workflow.add_edge("verify", END)

    return workflow.compile()


def main():
    parser = argparse.ArgumentParser(
        description="Autonomous Deep Research Agent with LangGraph"
    )
    parser.add_argument(
        "--query",
        type=str,
        default="State of multi-agent orchestration architectures in 2026",
        help="Research query to investigate",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=2,
        help="Maximum search/reflection cycles (default: 2)",
    )
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY") or not os.getenv("TAVILY_API_KEY"):
        print("[Warning] Ensure OPENAI_API_KEY and TAVILY_API_KEY are configured in .env", file=sys.stderr)

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
        "citation_audit": {},
    }

    print(f"\n[Agent] Starting research on: '{args.query}' (Max Cycles: {args.max_iterations})\n")

    # Stream node transitions for live CLI feedback
    for update in agent.stream(initial_state, stream_mode="updates"):
        for node_name, state_delta in update.items():
            if node_name == "decompose":
                queries = state_delta.get("sub_queries", [])
                print(f"[1/5 Decompose] Generated {len(queries)} sub-queries:")
                for q in queries:
                    print(f"  - {q}")
            elif node_name == "search":
                sources = state_delta.get("search_results", [])
                print(f"[2/5 Search] Total sources collected so far: {len(sources)}")
            elif node_name == "evaluate":
                notes = state_delta.get("critique_notes", "")
                is_suff = state_delta.get("is_sufficient", True)
                print(f"[3/5 Evaluate] Sufficient: {is_suff} | {notes}")
            elif node_name == "synthesize":
                print("[4/5 Synthesize] Draft report generated.")
            elif node_name == "verify":
                audit = state_delta.get("citation_audit", {})
                invalid = audit.get("invalid_indices", [])
                print(f"[5/5 Verify] Audit complete. Out-of-bounds citations: {invalid if invalid else 'None'}")
                print("\n" + "=" * 70)
                print(state_delta.get("verified_report", ""))
                print("=" * 70)


if __name__ == "__main__":
    main()