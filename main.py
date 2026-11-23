
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from graph.workflow import run_competitor_intelligence


# =====================================
# 1. INITIALIZE APPLICATION
# =====================================

app = FastAPI(
    title="Multi Agent Analysis Engine",
    description="Multi-Agent AI Competitor Research System",
    version="1.0.0"
)

# CORS Middleware
origins = [
    "http://localhost",
    "http://localhost:8080",
    "https://8080-firebase-py-ws-2-1780327883095.cluster-y75up3teuvc62qmnwys4deqv6y.cloudworkstations.dev",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =====================================
# 2. REQUEST DATA MODEL
# =====================================

class ResearchRequest(BaseModel):

    company: str = Field(min_length=1)

    competitors: list[str] = Field(min_length=1)

    country: str

    research_objective: str


# =====================================
# 3. HOME ENDPOINT
# =====================================

@app.get("/")
def home():

    return {
        "message": "Welcome to Multi Agent Analysis Engine using Langgraph",
        "status": "running"
    }


# =====================================
# 4. HEALTH CHECK ENDPOINT
# =====================================

@app.get("/health")
def health_check():

    return {
        "status": "healthy",
        "service": "Multi Agent Analysis Engine API"
    }


# =====================================
# 5. RESEARCH ENDPOINT
# =====================================

@app.post("/api/v1/research")
def conduct_research(request: ResearchRequest):

    result = run_competitor_intelligence(
        company=request.company,
        competitors=request.competitors,
        country=request.country,
        research_objective=request.research_objective
    )

    return {
        "success": True,
        "company": request.company,
        "competitors": request.competitors,
        "country": request.country,
        "report": result["final_report"]
    }