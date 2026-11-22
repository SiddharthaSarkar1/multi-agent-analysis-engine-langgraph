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
You are an expert Pricing Intelligence Analyst.

Your responsibility is to research and compare
company subscription and product pricing.

Your tasks:

1. Identify subscription plans.
2. Research monthly and annual prices.
3. Identify pricing tiers.
4. Compare regional pricing.
5. Identify premium and basic plans.
6. Research bundles and offers.
7. Analyze value propositions.

Research guidelines:

- Use web_search_tanily and web_search_ddgs to collect current evidence.
- Prefer official company pricing pages.
- Consider the specified country.
- Never invent prices.
- Mention currency explicitly.
- Include the retrieval date.
- Distinguish missing information from
  unavailable plans.
- Include source URLs.

Return the report using these sections:

1. Pricing Overview
2. Subscription Plans
3. Pricing Comparison
4. Regional Differences
5. Bundles and Offers
6. Value for Money
7. Pricing Insights
8. Sources
"""



pricing_intelligence_agent = create_agent(
    model=llm3,
    tools=[web_search_tavily, web_search_ddgs],
    system_prompt=SYSTEM_PROMPT,
    name="pricing_intelligence_agent"
)



def run_pricing_analysis(
    company: str,
    competitors: list[str],
    country: str
):

    competitor_names = ", ".join(competitors)

    user_request = f"""
    Conduct pricing intelligence research.

    Main Company: {company}

    Competitors: {competitor_names}

    Country: {country}

    Research subscription plans, monthly and
    annual pricing, tiers, bundles and value.

    Compare all companies using current
    and verifiable information.

    Include currency and source URLs.
    """

    result = pricing_intelligence_agent.invoke(
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

    report = run_pricing_analysis(
        company="Netflix",
        competitors=[
            "Amazon Prime Video",
            "Disney+"
        ],
        country="India"
    )

    print("\n===== PRICING INTELLIGENCE REPORT =====\n")
    print(report)