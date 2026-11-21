import os
from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv()

tavily = TavilyClient(
    os.getenv("TAVILY_API_KEY")
)

query = """
Netflix company overview, products,
streaming services and recent developments
"""

response = tavily.search(query=query, max_results=5)

print("=========TAVILY response==========")

for index, result in enumerate(response["results"], start=1):

    print(f"\nResult {index}")
    print("Title:", result["title"])
    print("URL:", result["url"])
    print("Content:", result["content"])
    print("-" * 60)