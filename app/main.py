from fastapi import FastAPI
from pydantic import BaseModel

from app.utilities import *
app = FastAPI()


class Question(BaseModel):
    question: str
    answer: str | None = None


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/question/")
async def create_item(question: Question):
    # print("Ask Your Question: ")
    # url : "https://klaus-api.web.app/question/"
    query = question.question
    if len(query) > 0:
        collection = generateVectorDatabase()
        llm = setupOpenAI()
        chain = setupLangchain(llm=llm)
        question.answer = main(query, collection=collection, chain=chain)
    else:
        question.answer = "Please ask your question."

    return question
