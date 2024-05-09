from fastapi import FastAPI
from pydantic import BaseModel
from app.models import ChatbotInput, ChatbotOutput
from Chatbot.chatbot import conversation

app = FastAPI()

@app.post("/chatbot/", response_model=ChatbotOutput)
async def generate_response(input_text: ChatbotInput):
    # Generate response using the input text
    response = conversation(
        {"question": f"{input_text}"}
    )
    return {"response": response["text"]}


@app.get("/chatbot/")
async def generate_response():
    # Generate response using the input text
    # response = conversation(
    #     {"question": f"{input_text}"}
    # )
    return {"Hi! How may I help you?"}