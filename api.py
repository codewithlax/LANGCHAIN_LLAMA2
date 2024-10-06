from fastapi import FastAPI
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
from langserve import add_routes
import uvicorn
import os
from dotenv import load_dotenv
load_dotenv()

# os.environ['LANGCHAIN_TRACING_V2'] = "true"
os.environ['LANGCHAIN_API_KEY'] = os.getenv("LANGCHAIN_API_KEY")

app = FastAPI (
    title="LLM chains using FlaskAPI's add_routes()",
    version= "1.0",
    description="Using FlaskAPI to serve LLMs to client side by adding routes (similar to chains + exposing endpoints)"
)

model = Ollama(model="llama2")
prompt_1 = ChatPromptTemplate.from_template("You are a chatbot. You have say something about {topic} an angry tone.")
prompt_2 = ChatPromptTemplate.from_template("You are a chatbot. You have say something about {topic} a funny tone.")

add_routes(
    app,
    prompt_1|model,
    path="/angry"

)

add_routes(
    app,
    prompt_2|model,
    path="/funny"

)

if __name__ == "__main__":
    uvicorn.run(app, host="localhost",port=8000)