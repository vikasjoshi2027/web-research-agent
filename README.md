# 🌐 Autonomous Deep Research Agent

An advanced, production-grade autonomous research agent built with **LangGraph**, **LangChain**, and **Tavily Search**, featuring a modern **Streamlit Web UI**.

The system autonomously decomposes research topics into multi-angle queries, conducts iterative web investigations with self-reflection, and executes a rigorous **Citation & Fact-Checking Audit** to ensure zero hallucinations.

**Author**: Vikas Joshi ([vikasjoshi.2027@gmail.com](mailto:vikasjoshi.2027@gmail.com))

---

## 🏗️ Agentic Architecture & Graph Workflow

```mermaid
flowchart TD
    Start([User Query]) --> Decompose[1. Query Decomposition]
    Decompose --> Search[2. Multi-Source Web Search]
    Search --> Evaluate[3. Reflection & Gap Analysis]
    
    Evaluate -- "Gaps Found (iteration < max)" --> FollowUp[Generate Follow-Up Queries]
    FollowUp --> Search
    
    Evaluate -- "Sufficient Evidence" --> Synthesize[4. Structured Draft Synthesis]
    Synthesize --> Verify[5. Fact-Checking & Citation Audit]
    Verify --> End([Final Verified Report & Audit Log])
```

---

## ⚡ Key Technical Innovations

1. **Autonomous Query Decomposition**: Breaks complex, ambiguous topics into multiple specialized sub-queries (background, technical breakthroughs, limitations).
2. **Cyclic Reflection & Gap Analysis (Self-Correction)**: The evaluator node audits retrieved evidence and dynamically triggers targeted follow-up searches if information gaps exist.
3. **URL Deduplication**: Automatically aggregates and deduplicates search findings across search cycles.
4. **Citation & Fact-Checking Verifier Node**: Cross-references every inline claim `[X]` against raw source snippets to eliminate LLM hallucinations.
5. **Streamlit Web Dashboard**: Real-time progress stepper, interactive source cards, model selector, and one-click Markdown export.

---

## 🚀 Quick Start

### 1. Prerequisites
- Python 3.10+
- [OpenAI API Key](https://platform.openai.com/api-keys)
- [Tavily Search API Key](https://app.tavily.com) (Free tier available)

### 2. Installation
```bash
git clone <your-repo-url>
cd web-research-agent
pip install -r requirements.txt
```

### 3. Configure API Keys
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Add your keys:
```env
OPENAI_API_KEY=your_openai_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
OPENAI_MODEL=gpt-4o-mini
```

---

## 🖥️ Running the Application

### Option A: Launch Interactive Web UI (Streamlit)
```bash
streamlit run app.py
```
*Opens an interactive browser dashboard with visual progress tracking, topic chips, source previews, and report downloads.*

### Option B: Run via Command-Line Interface (CLI)
```bash
# Run default investigation
python agent.py

# Run custom topic with 2 reflection cycles
python agent.py --query "State of multi-agent orchestration architectures in 2026" --max-iterations 2
```

---

## 📁 Project Structure

```
web-research-agent/
├── app.py              # Streamlit Web UI with real-time stepper & export
├── agent.py            # Main LangGraph cyclic state graph with 5 agentic nodes
├── requirements.txt    # Project dependencies
├── metadata.yaml       # Project configuration, tags, and author info
├── .env.example        # Environment variable template
├── .gitignore          # Git ignore rules
├── LICENSE             # MIT License
└── README.md           # Documentation & Architecture
```

---

## 🎯 Interview Highlights (Why This Project Stands Out)

- **Why LangGraph?** Leverages StateGraph to manage state across cyclic evaluation loops and conditional routing edges.
- **Hallucination Prevention**: Includes a dedicated adversarial verification node for source-claim alignment.
- **Bounded Recursion**: Uses `max_iterations` guards to guarantee deterministic termination and cost control.
- **Full-Stack AI**: Complete with both headless CLI execution and an interactive Streamlit UI.

---

## 👤 Author

- **Vikas Joshi** ([vikasjoshi.2027@gmail.com](mailto:vikasjoshi.2027@gmail.com))

## 📄 License

This project is licensed under the [MIT License](LICENSE).