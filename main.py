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
# Groq model initialization
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
    tasks: List[str] = []
    estimated_tasks: List[Dict] = []
    final_plan: List[Dict] = []

class TaskList(BaseModel):
    tasks: List[str] = Field(description="A list of distinct project tasks")

class TaskEstimation(BaseModel):
    task_name: str
    estimated_hours: int
    rationale: str

class EstimationList(BaseModel):
    estimations: List[TaskEstimation]

# --- 3. NODE DEFINITIONS (Fixed for Subscriptability) ---

def planner_node(state: Any):
    # Convert Pydantic object to dict if necessary to prevent subscriptable error
    s = state if isinstance(state, dict) else state.dict()
    
    print(f"--- PLANNER: Processing {s.get('project_name')} ---")
    parser = JsonOutputParser(pydantic_object=TaskList)
    prompt = ChatPromptTemplate.from_template(
        "Planner in {industry}. Project: {project_name}. Req: {requirements}. List tasks. {format_instructions}"
    )
    chain = prompt | llm | parser
    result = chain.invoke({
        "industry": s.get("industry"),
        "project_name": s.get("project_name"),
        "requirements": s.get("requirements"),
        "format_instructions": parser.get_format_instructions()
    })
    return {"tasks": result["tasks"]}

def estimator_node(state: Any):
    s = state if isinstance(state, dict) else state.dict()
    print("--- ESTIMATOR WORKING ---")
    parser = JsonOutputParser(pydantic_object=EstimationList)
    tasks_str = "\n".join(s.get("tasks", []))
    prompt = ChatPromptTemplate.from_template("Estimate hours for: {tasks_str}. {format_instructions}")
    chain = prompt | llm | parser
    result = chain.invoke({"tasks_str": tasks_str, "format_instructions": parser.get_format_instructions()})
    return {"estimated_tasks": result["estimations"]}

def allocator_node(state: Any):
    s = state if isinstance(state, dict) else state.dict()
    print("--- ALLOCATOR WORKING ---")
    team_str = str(s.get("team_members", []))
    tasks_data = str(s.get("estimated_tasks", []))
    prompt = ChatPromptTemplate.from_template(
        "Assign tasks: {tasks_data} to Team: {team_str}. "
        "Return a JSON list of objects with keys: 'task_name', 'assigned_to', 'estimated_hours', 'rationale'."
    )
    chain = prompt | llm | JsonOutputParser()
    result = chain.invoke({"team_str": team_str, "tasks_data": tasks_data})
    
    plan = result if isinstance(result, list) else result.get("allocations", [])
    return {"final_plan": plan}

# --- 4. GRAPH CONSTRUCTION ---
workflow = StateGraph(ProjectState)
workflow.add_node("planner", planner_node)
workflow.add_node("estimator", estimator_node)
workflow.add_node("allocator", allocator_node)

workflow.set_entry_point("planner")
workflow.add_edge("planner", "estimator")
workflow.add_edge("estimator", "allocator")
workflow.add_edge("allocator", END)

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
        # Initialize the state from the incoming request
        initial_state = request.model_dump()
        # Run the LangGraph
        result = runner.invoke(initial_state)
        return {
            "project": result.get("project_name") if isinstance(result, dict) else result.project_name,
            "plan": result.get("final_plan") if isinstance(result, dict) else result.final_plan
        }
    except Exception as e:
        print(f"Server Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
