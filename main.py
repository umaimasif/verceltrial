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
# The GROQ_API_KEY must be set in your Vercel Environment Variables
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

# --- 3. NODE DEFINITIONS ---

def planner_node(state: Any):
    s = state if isinstance(state, dict) else state.dict()
    parser = JsonOutputParser(pydantic_object=TaskList)
    prompt = ChatPromptTemplate.from_template(
        "You are a Project Planner. Project: {project_name}. Industry: {industry}. "
        "Requirements: {requirements}. Break this into a list of tasks. {format_instructions}"
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
    parser = JsonOutputParser(pydantic_object=EstimationList)
    tasks_str = "\n".join(s.get("tasks", []))
    prompt = ChatPromptTemplate.from_template(
        "Estimate realistic hours for these tasks: {tasks_str}. {format_instructions}"
    )
    chain = prompt | llm | parser
    result = chain.invoke({"tasks_str": tasks_str, "format_instructions": parser.get_format_instructions()})
    return {"estimated_tasks": result["estimations"]}

def allocator_node(state: Any):
    s = state if isinstance(state, dict) else state.dict()
    team_str = str(s.get("team_members", []))
    tasks_data = str(s.get("estimated_tasks", []))
    
    # --- STRICT JSON PROMPT ---
    prompt = ChatPromptTemplate.from_template(
        "You are a Resource Allocator. Assign these tasks: {tasks_data} to this team: {team_str}. "
        "Match tasks to skills. "
        "CRITICAL: Output ONLY a valid JSON list of objects. No intro text, no python code, no markdown backticks. "
        "Format: [{{'task_name': '...', 'assigned_to': '...', 'estimated_hours': 0, 'rationale': '...'}}]"
    )
    
    # We use a standard JsonOutputParser to enforce structure
    chain = prompt | llm | JsonOutputParser()
    
    try:
        result = chain.invoke({"team_str": team_str, "tasks_data": tasks_data})
        # Handle cases where AI returns a dict with a 'plan' key instead of a raw list
        plan = result if isinstance(result, list) else result.get("allocations", result.get("plan", []))
        return {"final_plan": plan}
    except Exception as e:
        print(f"Allocation Error: {e}")
        raise
