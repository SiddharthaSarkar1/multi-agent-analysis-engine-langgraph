import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama

load_dotenv()

model_name = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

# Initialize the llm'set

llm1 = ChatGroq(model=model_name, temperature=0.2, max_retries=2)

# response1 = llm1.invoke(
#     "Explain what competitor intelligence means in business " "using a simple example."
# )

# print("=========LLM-1 response==========")
# print(response1.content)

ollama_remote_base_url = "https://reentry-subject-mannish.ngrok-free.dev"

llm2 = ChatOllama(
    model="qwen3.5:9b", temperature=0.2, max_retries=2, base_url=ollama_remote_base_url
)

llm3 = ChatOllama(
    model="llama3.2:3b", temperature=0.2, max_retries=2, base_url=ollama_remote_base_url
)

response2 = llm2.invoke(
    "Explain what competitor intelligence means in business " "using a simple example."
)

print("=========LLM-2 response==========")
print(response2.content)

response3 = llm3.invoke(
    "Explain what competitor intelligence means in business " "using a simple example."
)

print("=========LLM-3 response==========")
print(response3.content)
