import os
import joblib
from typing import TypedDict

from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI

from dotenv import load_dotenv


load_dotenv()


GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

# --- CONFIG ---
model_pipeline = joblib.load('admission_pipeline.pkl')
features = joblib.load('feature_names.pkl')

# --- LANGGRAPH LOGIC ---
class AgentState(TypedDict):
    input_data: dict
    prediction: int
    probability: float
    explanation: str

def gemini_counselor(state: AgentState):
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        api_key=GEMINI_API_KEY,
        temperature=0.5
    )

    status = "Admitted" if state['prediction'] == 1 else "Not Admitted"
    
    prompt = f"""
    Context: You are an AI Academic Advisor for FlexiSAF. 
    Student Data: {state['input_data']}
    Result: {status} (Confidence: {state['probability']}%)
    
    Instruction: Write a 3-sentence professional explanation. 
    Explain how their CGPA and GRE scores specifically affected this outcome.
    """
    response = llm.invoke(prompt)
    return {"explanation": response.content}

# Compile Graph
workflow = StateGraph(AgentState)
workflow.add_node("advise", gemini_counselor)
workflow.set_entry_point("advise")
workflow.add_edge("advise", END)
graph = workflow.compile()
