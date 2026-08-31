# 🌐 Web Research Agent

An intelligent autonomous AI agent built with **LangGraph**, **LangChain**, and **Tavily Search** that investigates any topic on the web and synthesizes a structured, cited research report.

**Author**: Vikas Joshi (<vikasjoshi.2027@gmail.com>)

---

## ⚡ Features

- **Autonomous Web Search**: Retrieves relevant web pages, news, and technical articles using Tavily Search API.
- **Context Synthesis**: Distills multiple web sources into an executive summary, key findings, and data-backed insights.
- **Source Citation**: Automatically lists referenced web sources and URLs for traceability.
- **LangGraph StateGraph**: Built with a modular graph architecture for state management and easy extensibility.

---

## 🛠️ Architecture

```
User Query ──> [ Search Node (Tavily) ] ──> [ Synthesis Node (LLM) ] ──> Final Research Report
```

---

## 🚀 Quick Start

### 1. Prerequisites
- Python 3.10+
- [OpenAI API Key](https://platform.openai.com/api-keys)
- [Tavily Search API Key](https://app.tavily.com) (Free tier available)

### 2. Installation
Clone your repository and navigate to the project directory:

```bash
git clone <your-repo-url>
cd web-research-agent
```

Install dependencies:
```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables
Copy the `.env.example` file to `.env` and fill in your API keys:

```bash
cp .env.example .env
```

Edit `.env`:
```env
OPENAI_API_KEY=your_openai_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
```

### 4. Run the Agent

Run with the default query:
```bash
python agent.py
```

Run with a custom query:
```bash
python agent.py --query "Latest advancements in agentic AI architectures"
```

---

## 📁 Project Structure

```
web-research-agent/
├── agent.py            # Main LangGraph agent logic & entrypoint
├── requirements.txt    # Project dependencies
├── metadata.yaml       # Project configuration and tags
├── .env.example        # Environment variable template
├── .gitignore          # Git ignore rules
├── LICENSE             # MIT License
└── README.md           # Documentation
```

---

## 👤 Author

- **Vikas Joshi** ([vikasjoshi.2027@gmail.com](mailto:vikasjoshi.2027@gmail.com))

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).