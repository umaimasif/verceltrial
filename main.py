import os
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langgraph.graph import StateGraph, END

# Initialize FastAPI
app = FastAPI()

# --- 1. CONFIGURATION ---
llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0)

# --- 2. DATA MODELS ---
class ProjectRequest(BaseModel):
    project_name: str
    requirements: str
    industry: str
    team_members: List[Dict[str, str]]

class ProjectState(BaseModel):
    project_name: str
    requirements: str
    industry: str
    team_members: List[Dict[str, str]]
    final_plan: List[Dict] = []

# --- 3. THE FAST UNIFIED AGENT NODE ---
def fast_agent_node(state: Any):
    # Ensure state is handled as a dict
    s = state if isinstance(state, dict) else state.dict()
    
    print(f"--- AGENT: Processing {s.get('project_name')} ---")
    
    # Unified prompt to do everything in one step (Fast & Reliable)
    prompt = ChatPromptTemplate.from_template(
        """
        You are an expert Project Manager. 
        Project: {project_name} ({industry})
        Requirements: {requirements}
        Team: {team_str}

        TASK:
        1. Break the project into 5-8 key tasks.
        2. Estimate realistic hours for each task.
        3. Assign each task to the best team member based on their skills.

        CRITICAL: Output ONLY a valid JSON list of objects. 
        No intro text, no markdown code blocks (```), no explanations.
        
        Format:
        [
          {{"task_name": "...", "assigned_to": "...", "estimated_hours": 0, "rationale": "..."}},
          {{"task_name": "...", "assigned_to": "...", "estimated_hours": 0, "rationale": "..."}}
        ]
        """
    )
    
    chain = prompt | llm | JsonOutputParser()
    
    try:
        # Run the AI
        result = chain.invoke({
            "project_name": s.get("project_name"),
            "industry": s.get("industry"),
            "requirements": s.get("requirements"),
            "team_str": str(s.get("team_members", []))
        })
        
        # Handle the result whether it's a raw list or a dict wrapper
        plan = result if isinstance(result, list) else result.get("plan", result.get("allocations", []))
        return {"final_plan": plan}
        
    except Exception as e:
        print(f"Agent Error: {e}")
        # In case of error, raise it so we see it in logs
        raise e

# --- 4. GRAPH CONSTRUCTION ---
workflow = StateGraph(ProjectState)
workflow.add_node("agent", fast_agent_node)
workflow.set_entry_point("agent")
workflow.add_edge("agent", END)
runner = workflow.compile()

# --- 5. ROUTES ---

# Route 1: The Main Home Page
@app.get("/", response_class=HTMLResponse)
async def read_root():
    try:
        with open("index.html", "r") as f:
            return f.read()
    except Exception as e:
        return f"<h1>Error: index.html not found</h1><p>{str(e)}</p>"

# Route 2: The Specific File (FIXES YOUR "NOT FOUND" ERROR)
@app.get("/index.html", response_class=HTMLResponse)
async def read_index_file():
    return await read_root()

# Route 3: The API Endpoint
@app.post("/api/generate-plan")
async def generate_plan(request: ProjectRequest):
    try:
        initial_state = request.model_dump()
        result = runner.invoke(initial_state)
        
        return {
            "project": result.get("project_name") if isinstance(result, dict) else result.project_name,
            "plan": result.get("final_plan") if isinstance(result, dict) else result.final_plan
        }
    except Exception as e:
        print(f"API Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
