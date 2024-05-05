from fastapi import FastAPI
from Chatbot.chatbot import conversation

app = FastAPI()

@app.post("/generate/")
async def generate_response(input_text: str):
    response = conversation(
        {"question": f"{input_text}"}
    )
    return {"response": response["text"]}