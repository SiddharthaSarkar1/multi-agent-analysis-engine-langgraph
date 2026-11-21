from ddgs import DDGS

query = """
google company overview, products,
streaming services and recent developments
"""

def web_search(query: str) -> str:
  """Search DuckDuckGo for recent information on a given query."""
  with DDGS() as ddgs:
    results = list(ddgs.text(query, max_results=5))

  if not results:
    return "No results found."

  formatted = []
  for idx, r in enumerate(results, start=1):
    formatted.append(f"\nResult: {idx} \nTitle: {r['title']}\nURL: {r['href']}\nSnippet: {r['body']}")
  return "\n\n".join(formatted)



print(web_search(query))