import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.web_search import web_search_tavily, web_search_ddgs
from dotenv import load_dotenv

from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from langchain.agents import create_agent

load_dotenv()

model_name = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

llm1 = ChatGroq(model=model_name, temperature=0.2, max_retries=2)

ollama_remote_base_url = "https://reentry-subject-mannish.ngrok-free.dev"

llm2 = ChatOllama(
    model="qwen3.5:9b", temperature=0.2, max_retries=2, base_url=ollama_remote_base_url
)

llm3 = ChatOllama(
    model="gemma4:latest", temperature=0.2, max_retries=2, base_url=ollama_remote_base_url
)

SYSTEM_PROMPT = """
You are an expert Product Analysis Agent
working in a Competitor Intelligence System.

Your responsibility is to compare products
and services offered by different companies.

Your tasks:

1. Research the products offered by each company.
2. Identify important product features.
3. Compare features across competitors.
4. Identify unique product capabilities.
5. Identify product strengths and limitations.
6. Analyze product differentiation.
7. Identify potential product gaps.

Research guidelines:

- Use web_search_tanily and web_search_ddgs for current product information.
- Prefer official company websites and documentation.
- Search for each company independently.
- Do not invent product features.
- Do not assume that unavailable information
  means a feature does not exist.
- Distinguish confirmed facts from analysis.
- Include source URLs for important findings.

Return your final analysis using these sections:

1. Product Overview
2. Feature Comparison
3. Unique Features
4. Product Strengths
5. Product Limitations
6. Competitive Differentiation
7. Product Opportunities
8. Sources
"""
product_analysis_agent = create_agent(
    model=llm3,
    tools=[web_search_tavily, web_search_ddgs],
    system_prompt=SYSTEM_PROMPT,
    name="product_analysis_agent",
)


def run_product_analysis(company: str, competitors: list[str]):

    competitors_text = ", ".join(competitors)

    user_request = f"""
    Conduct a detailed product comparison.

    Main Company: {company}

    Competitors: {competitors_text}

    Compare their products, features,
    differentiators, advantages and limitations.

    Use web search to collect evidence.

    Return a structured comparison report.
    """

    result = product_analysis_agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": user_request,
                }
            ]
        }
    )

    return result["messages"][-1].content


if __name__ == "__main__":

    report = run_product_analysis(
        company="Netflix", competitors=["Amazon Prime Video", "Disney+"]
    )

    print("\n===== PRODUCT ANALYSIS REPORT =====\n")
    print(report)
