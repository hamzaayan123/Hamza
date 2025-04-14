from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

chatbot = pipeline("text-generation", model="EleutherAI/gpt-neo-1.3B")

class ChatInput(BaseModel):
    message: str

@app.post("/chat")
async def chat_with_aayat(input: ChatInput):
    prompt = input.message
    reply = chatbot(prompt, max_length=100, do_sample=True, top_k=50)[0]["generated_text"]
    return {"response": reply}
