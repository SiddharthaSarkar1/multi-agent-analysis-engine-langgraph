import os
import json

from dotenv import load_dotenv
from tavily import TavilyClient
from ddgs import DDGS
from langchain_core.tools import tool

load_dotenv()

tavily_client = TavilyClient(os.getenv("TAVILY_API_KEY"))


@tool
def web_search_tavily(query: str) -> str:
    """
    Search the web for company and market information using tavily Search. Return concise research evidence.
    """

    try:
        response = tavily_client.search(query=query, max_results=3, search_depth="basic")

        results = []

        for item in response.get("results", []):

            results.append(
                {
                    "title": item.get("title"),
                    "url": item.get("url"),
                    "content": (item.get("content") or "")[:200],
                }
            )

        return json.dumps(results, indent=2)

    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def web_search_ddgs(query: str) -> str:
    """
    Search the web for company and market information using duckduckgo search. Return concise research evidence.
    """

    try:
        with DDGS() as ddgs:
            response = list(ddgs.text(query, max_results=5))

        results = []

        for index, result in enumerate(response, start=1):

            results.append(
                {
                    "Result": index,
                    "Title": result["title"],
                    "URL": result["href"],
                    "Content": result["body"]
                }
            )

        return json.dumps(results, indent=2)

    except Exception as e:
        return json.dumps({"error": str(e)})


#=====================
# =====Test======
#=====================

# response1 = web_search_tavily.invoke("google company overview, products, streaming services and recent developments")
# response2 = web_search_ddgs.invoke("google company overview, products, streaming services and recent developments")

# print(response1)
# print(response2)