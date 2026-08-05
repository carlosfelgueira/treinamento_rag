import warnings
warnings.filterwarnings("ignore")

from fastapi import FastAPI
from pydantic import BaseModel
from src.rag import ask

class Item(BaseModel):
    query: str

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "treinamento_rag API"}

@app.post("/query")
async def query(item: Item):
    return ask(item.query)
