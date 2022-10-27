from fastapi import FastAPI
from pydantic import BaseModel


from typing import List
import datetime
app = FastAPI()

class Cell(BaseModel):
    temp: float
    pm2_5: float
    pm10: float

class MapMatrix(BaseModel):
    x: int
    y: int
    k: float
    matrix: List[Cell]

class ResponseMatrix(BaseModel):
    x: int
    y: int
    k: float
    snapshot: datetime.datetime
    matrix: List[Cell]

@app.post("/matrix/")
async def push_matrix(matrix: MapMatrix) -> MapMatrix:
    return matrix

MOCKED_MATRIX = ResponseMatrix(
    snapshot= datetime.datetime(2022, 10, 27, 12, 25, 0),
    x=10,
    y=10,
    k=1,
    matrix=[Cell(temp=0+0.5*i, pm2_5=(10+0.1*i), pm10=(25+0.1*i)) for i in range(100)]
)

@app.get("/matrix/{snapshot}")
async def get_matrix(snapshot: str) -> ResponseMatrix:
    return MOCKED_MATRIX


@app.get("/")
async def root():
    return {"message": "Hello World"}

