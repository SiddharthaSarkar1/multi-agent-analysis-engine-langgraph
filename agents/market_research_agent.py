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
    model="llama3.2:3b", 
    temperature=0.2, 
    max_retries=2, 
    base_url=ollama_remote_base_url
)


SYSTEM_PROMPT = """
You are an expert Market Research Analyst
working for a Competitive Intelligence System.

Your responsibility is to research companies
and their market position.

Your tasks:

1. Research company background.
2. Identify the industry and business model.
3. Research market trends.
4. Identify major competitors.
5. Find recent company developments.
6. Collect relevant evidence.

Instructions:

- Use the web_search tool for current information.
- Search using specific and meaningful queries.
- Never invent facts, statistics or market figures.
- Clearly identify missing information.
- Distinguish verified facts from analysis.
- Include source URLs for important findings.

Return your final research in these sections:

1. Company Overview
2. Industry Overview
3. Business Model
4. Market Position
5. Major Competitors
6. Recent Developments
7. Key Findings
8. Sources
"""

# Langchain AI Agent

market_research_agent = create_agent(
    model=llm2,
    tools=[web_search_tavily, web_search_ddgs],
    system_prompt=SYSTEM_PROMPT,
    name="market_research_agent"
)

def run_market_research(company: str):

    user_request = f"""
    Conduct comprehensive market research
    on the following company:

    Company: {company}

    Research its background, industry,
    business model, market position,
    competitors and recent developments.

    Use web search to collect evidence.
    """

    response = market_research_agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": user_request,
                }
            ]
        }
    )

    final_response = response["messages"][-1]

    return final_response.content


# Test

if __name__ == "__main__":

    company_name = "Amazon"
    response = run_market_research(company_name)

    print("\n========== MARKET RESEARCH REPORT ==========\n")
    print(response)
