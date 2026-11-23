import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dotenv import load_dotenv

from pydantic import BaseModel, Field

from langchain_ollama import ChatOllama


load_dotenv()

# ----------------------------------
# STEP 1: DEFINE OUTPUT SCHEMA
# ----------------------------------

class ProductAnalysis(BaseModel):

    company_name: str = Field(
        description="Name of the company"
    )

    products: list[str] = Field(
        description="Important products offered by the company"
    )

    features: list[str] = Field(
        description="Important product features"
    )

    strengths: list[str] = Field(
        description="Product strengths"
    )

    weaknesses: list[str] = Field(
        description="Product weaknesses"
    )

    competitive_advantages: list[str] = Field(
        description="Competitive advantages of the products"
    )


# ----------------------------------
# STEP 2: INITIALIZE GEMINI
# ----------------------------------

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


# ----------------------------------
# STEP 3: ENABLE STRUCTURED OUTPUT
# ----------------------------------

structured_llm = llm3.with_structured_output(
    ProductAnalysis
)


# ----------------------------------
# STEP 4: SEND USER QUERY
# ----------------------------------

response = structured_llm.invoke(
    """
    Analyze Netflix as a company.

    Identify its important products, product features,
    strengths, weaknesses and competitive advantages.

    Return factual and concise information.
    Do not invent specific facts.
    """
)


# ----------------------------------
# STEP 5: DISPLAY STRUCTURED RESULT
# ----------------------------------

print("\nSTRUCTURED PRODUCT ANALYSIS\n")

print(response.model_dump_json(indent=4))


# ----------------------------------
# STEP 6: ACCESS INDIVIDUAL FIELDS
# ----------------------------------

print("\nCOMPANY:", response.company_name)

print("\nPRODUCTS:")

for product in response.products:
    print("-", product)
