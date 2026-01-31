import os
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langgraph.graph import StateGraph, END

# Initialize FastAPI
app = FastAPI()

# --- 1. SETUP & CONFIGURATION ---
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

# --- 3. THE UNIFIED FAST NODE ---

def fast_agent_node(state: Any):
    # Ensure state is handled as a dict
    s = state if isinstance(state, dict) else state.dict()
    
    print(f"--- UNIFIED AGENT: Planning project {s.get('project_name')} ---")
    
    # We combine all steps into one prompt to save time
    prompt = ChatPromptTemplate.from_template(
        """
        You are a Master Project Manager. 
        Project: {project_name} ({industry})
        Requirements: {requirements}
        Team: {team_str}

        TASK:
        1. Create 5-7 key tasks to build this project.
        2. Estimate realistic hours for each task.
        3. Assign each task to the best team member based on their skills.
        
        CRITICAL: Your response must be ONLY a raw JSON list. 
        No intro, no code blocks, no backticks.
        Format: [
          {{"task_name": "...", "assigned_to": "...", "estimated_hours": 0, "rationale": "..."}}
        ]
        """
    )
    
    chain = prompt | llm | JsonOutputParser()
    
    try:
        result = chain.invoke({
            "project_name": s.get("project_name"),
            "industry": s.get("industry"),
            "requirements": s.get("requirements"),
            "team_str": str(s.get("team_members", []))
        })
        
        # Extract list regardless of AI wrapper
        plan = result if isinstance(result, list) else result.get("plan", result.get("allocations", []))
        return {"final_plan": plan}
    except Exception as e:
        print(f"Fast Agent Error: {e}")
        raise e

# --- 4. GRAPH CONSTRUCTION (SIMPLIFIED) ---
workflow = StateGraph(ProjectState)
workflow.add_node("agent", fast_agent_node)
workflow.set_entry_point("agent")
workflow.add_edge("agent", END)

runner = workflow.compile()

# --- 5. ROUTES ---

@app.get("/", response_class=HTMLResponse)
async def read_index():
    try:
        # Serves the index.html from the root directory
        with open("index.html", "r") as f:
            return f.read()
    except Exception as e:
        return f"<h1>Error: index.html not found</h1><p>{str(e)}</p>"

@app.post("/api/generate-plan")
async def generate_plan(request: ProjectRequest):
    try:
        initial_state = request.model_dump()
        result = runner.invoke(initial_state)
        
        return {
            "project": result.get("project_name") if isinstance(result, dict) else result.project_name,
            "plan": result
