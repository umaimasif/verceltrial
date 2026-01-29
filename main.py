import os
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langgraph.graph import StateGraph, END

# Initialize FastAPI
app = FastAPI()

# --- 1. SETUP & CONFIGURATION ---
# On Vercel, GROQ_API_KEY will come from the system environment variables automatically.
# We do not hardcode the key here for security.

llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0)

# --- 2. DATA MODELS (Input/Output) ---

# This model defines what the User sends to your API
class ProjectRequest(BaseModel):
    project_name: str
    requirements: str
    industry: str
    team_members: List[Dict[str, str]]

# Internal State Dictionary
class ProjectState(BaseModel):
    project_name: str
    requirements: str
    industry: str
    team_members: List[Dict[str, str]]
    tasks: List[str] = []
    estimated_tasks: List[Dict] = []
    final_plan: List[Dict] = []

# --- 3. Pydantic Models for Agents ---
class TaskList(BaseModel):
    tasks: List[str] = Field(description="A list of distinct project tasks")

class TaskEstimation(BaseModel):
    task_name: str
    estimated_hours: int
    rationale: str

class EstimationList(BaseModel):
    estimations: List[TaskEstimation]

class TaskAllocation(BaseModel):
    task_name: str
    assigned_to: str
    estimated_hours: int
    rationale: str

class AllocationList(BaseModel):
    allocations: List[TaskAllocation]

# --- 4. NODE DEFINITIONS ---

def planner_node(state: dict):
    # Note: State comes in as a dict in LangGraph compilation
    print(f"--- PLANNER: Processing {state['project_name']} ---")
    parser = JsonOutputParser(pydantic_object=TaskList)
    prompt = ChatPromptTemplate.from_template(
        """
        You are an expert Project Planner in the {industry} industry.
        Project Name: {project_name}
        Requirements: {requirements}
        Goal: Break this project down into a logical Work Breakdown Structure (WBS).
        List the key tasks.
        {format_instructions}
        """
    )
    chain = prompt | llm | parser
    result = chain.invoke({
        "industry": state["industry"],
        "project_name": state["project_name"],
        "requirements": state["requirements"],
        "format_instructions": parser.get_format_instructions()
    })
    return {"tasks": result["tasks"]}

def estimator_node(state: dict):
    print("--- ESTIMATOR WORKING ---")
    parser = JsonOutputParser(pydantic_object=EstimationList)
    tasks_str = "\n".join([f"- {t}" for t in state["tasks"]])
    prompt = ChatPromptTemplate.from_template(
        """
        You are an expert Estimation Analyst.
        Task List:
        {tasks_str}
        Estimate the effort (hours) for each task. Be realistic.
        {format_instructions}
        """
    )
    chain = prompt | llm | parser
    result = chain.invoke({
        "tasks_str": tasks_str,
        "format_instructions": parser.get_format_instructions()
    })
    return {"estimated_tasks": result["estimations"]}

def allocator_node(state: dict):
    print("--- ALLOCATOR WORKING ---")
    parser = JsonOutputParser(pydantic_object=AllocationList)
    team_str = "\n".join([f"Name: {m['name']}, Skills: {m['skills']}" for m in state["team_members"]])
    tasks_data = state["estimated_tasks"]
    prompt = ChatPromptTemplate.from_template(
        """
        You are an expert Resource Allocator.
        Team:
        {team_str}
        Tasks (with estimates):
        {tasks_data}
        Assign tasks to the most suitable team member based on skills.
        If skills don't match, assign to 'Unassigned'.
        {format_instructions}
        """
    )
    chain = prompt | llm | parser
    result = chain.invoke({
        "team_str": team_str,
        "tasks_data": tasks_data,
        "format_instructions": parser.get_format_instructions()
    })
    return {"final_plan": result["allocations"]}

# --- 5. GRAPH CONSTRUCTION ---
workflow = StateGraph(ProjectState) # Use the Pydantic model for type hint if desired, or simple dict
workflow.add_node("planner", planner_node)
workflow.add_node("estimator", estimator_node)
workflow.add_node("allocator", allocator_node)

workflow.set_entry_point("planner")
workflow.add_edge("planner", "estimator")
workflow.add_edge("estimator", "allocator")
workflow.add_edge("allocator", END)

runner = workflow.compile()

# --- 6. API ENDPOINT ---

@app.get("/")
def home():
    return {"message": "Project AI Agent API is running"}

@app.post("/api/generate-plan")
async def generate_plan(request: ProjectRequest):
    try:
        # Convert Pydantic request to dict for LangGraph
        initial_state = request.model_dump()
        initial_state["tasks"] = []
        initial_state["estimated_tasks"] = []
        initial_state["final_plan"] = []

        result = runner.invoke(initial_state)
        
        return {
            "project": result["project_name"],
            "plan": result["final_plan"]
        }
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))