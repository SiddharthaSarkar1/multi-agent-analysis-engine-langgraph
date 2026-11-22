#  Strengths, Weaknesses, Opportunities, and Threats for a business

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tools.web_search import web_search_tavily, web_search_ddgs
from dotenv import load_dotenv

from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from langchain.agents import create_agent

load_dotenv()

model_name = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

llm1 = ChatGroq(
    model=model_name, 
    temperature=0.2, 
    max_retries=2
)

ollama_remote_base_url = "https://reentry-subject-mannish.ngrok-free.dev"

llm2 = ChatOllama(
    model="qwen3.5:9b", 
    temperature=0.2, 
    max_retries=2, 
    base_url=ollama_remote_base_url
)

llm3 = ChatOllama(
    model="gemma4:latest", 
    temperature=0.2, 
    max_retries=2, 
    base_url=ollama_remote_base_url
)

SYSTEM_PROMPT = """
You are a Senior Business Strategy Analyst
specializing in SWOT analysis.

Your responsibility is to analyze companies
and identify strategic business factors.

Your tasks:

1. Identify company strengths.
2. Identify company weaknesses.
3. Identify market opportunities.
4. Identify external threats.
5. Analyze competitive positioning.
6. Recommend strategic improvements.

Research guidelines:

- Use web_search_tavily and web_search_ddgs for relevant evidence.
- Prefer reliable and official sources.
- Use the provided research reports.
- Distinguish internal and external factors.
- Never invent company facts.
- Do not present assumptions as facts.
- Explain the reasoning behind each finding.
- Include source URLs where possible.

Return the report using these sections:

1. Company Overview
2. Strengths
3. Weaknesses
4. Opportunities
5. Threats
6. Competitive Risks
7. Strategic Recommendations
8. Sources
"""


swot_analysis_agent = create_agent(
    model=llm3,
    tools=[web_search_tavily, web_search_ddgs],
    system_prompt=SYSTEM_PROMPT,
    name="swot_analysis_agent"
)


def run_swot_analysis(
    company: str,
    competitors: list[str],
    research_context: str = ""
):

    competitor_names = ", ".join(competitors)

    user_request = f"""
    Perform a comprehensive SWOT analysis.

    Company: {company}

    Competitors: {competitor_names}

    Existing Research Findings:
    {research_context}

    Analyze the company's internal strengths
    and weaknesses and its external opportunities
    and threats.

    Use web search to verify relevant facts.

    Provide evidence-backed strategic
    recommendations.
    """

    result = swot_analysis_agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": user_request
                }
            ]
        }
    )

    return result["messages"][-1].content


# Independent testing
if __name__ == "__main__":

    report = run_swot_analysis(
        company="Netflix",
        competitors=[
            "Amazon Prime Video",
            "Disney+"
        ],
        research_context=""
    )

    print("\n===== SWOT ANALYSIS REPORT =====\n")
    print(report)