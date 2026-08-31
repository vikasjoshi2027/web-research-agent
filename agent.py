"""
Web Research Agent using LangGraph + Tavily Search.

An automated research assistant that accepts an investigation query, executes web searches,
and synthesizes findings into a comprehensive structured report with cited references.

Author: Vikas Joshi <vikasjoshi.2027@gmail.com>

Usage:
    python agent.py
    python agent.py --query "Latest advancements in agentic AI architectures"
"""

import argparse
import os
import sys
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

# Load environment variables
load_dotenv()


class ResearchState(TypedDict):
    """State definition for the web research graph."""
    messages: Annotated[list, add_messages]
    query: str
    search_results: list[dict]
    report: str


def search_web(state: ResearchState) -> dict:
    """Queries Tavily search to retrieve context on the topic."""
    query = state["query"]
    print(f"🔍 Searching the web for: '{query}'...")

    try:
        tool = TavilySearch(max_results=5)
        raw_results = tool.invoke(query)

        if isinstance(raw_results, dict):
            results = raw_results.get("results", [])
        elif isinstance(raw_results, list):
            results = raw_results
        else:
            results = []
    except Exception as exc:
        print(f"⚠️ Search failed or encountered an issue: {exc}")
        results = []

    print(f"✅ Retrieved {len(results)} search results.")
    return {"search_results": results}


def synthesize_report(state: ResearchState) -> dict:
    """Synthesizes retrieved search results into a clean markdown report."""
    query = state["query"]
    search_results = state.get("search_results", [])

    print("📝 Synthesizing research report...")

    # Format search context
    context_lines = []
    for idx, result in enumerate(search_results, 1):
        title = result.get("title", "Untitled")
        url = result.get("url", "N/A")
        content = result.get("content", "")
        context_lines.append(f"[{idx}] {title}\nURL: {url}\nContent: {content}\n")

    context = "\n".join(context_lines) if context_lines else "No search results retrieved."

    prompt = f"""You are an expert research analyst. Based on the web search results provided below, write a comprehensive, authoritative, and well-structured research report.

Topic: {query}

Search Results:
{context}

Format the report with the following structure:
# Research Report: {query}

## Executive Summary
(A concise, high-level summary of the findings)

## Key Findings & Detailed Analysis
(In-depth points, trends, technical details, and breakdown)

## Sources & Citations
(Numbered list of sources consulted with title and URL)
"""

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    response = llm.invoke([
        SystemMessage(content="You are an expert technical researcher and synthesizer."),
        HumanMessage(content=prompt)
    ])

    return {
        "report": response.content,
        "messages": [response]
    }


def build_research_graph():
    """Builds and compiles the LangGraph StateGraph workflow."""
    workflow = StateGraph(ResearchState)

    # Register nodes
    workflow.add_node("search", search_web)
    workflow.add_node("synthesize", synthesize_report)

    # Configure graph edges
    workflow.set_entry_point("search")
    workflow.add_edge("search", "synthesize")
    workflow.add_edge("synthesize", END)

    return workflow.compile()


def main():
    parser = argparse.ArgumentParser(description="Web Research Agent")
    parser.add_argument(
        "--query",
        type=str,
        default="State of multimodal AI and autonomous agents",
        help="Topic or question to research",
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
        print("   Please add them to your .env file.\n")

    agent = build_research_graph()

    initial_state = {
        "messages": [],
        "query": args.query,
        "search_results": [],
        "report": ""
    }

    output = agent.invoke(initial_state)

    print("\n" + "=" * 70)
    print("📋 FINAL RESEARCH REPORT")
    print("=" * 70 + "\n")
    print(output["report"])


if __name__ == "__main__":
    main()