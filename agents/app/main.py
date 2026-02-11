from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Optional
from app.graph import graph

app = FastAPI(title="booking-agent")

class ChatReq(BaseModel):
    messages: List[Dict[str, str]] = []
    text: str

class ChatResp(BaseModel):
    answer: str
    tool_result: Optional[dict] = None

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/chat", response_model=ChatResp)
def chat(req: ChatReq):
    state = {"messages": req.messages, "text": req.text, "intent": None, "tool_result": None, "answer": None}
    out = graph.invoke(state)
    return ChatResp(answer=out["answer"], tool_result=out.get("tool_result"))
