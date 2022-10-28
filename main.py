from fastapi import FastAPI


from models.MatrixModels import MapMatrix, ResponseMatrix
from services import simulator
import json

app = FastAPI()

@app.post("/matrix/")
async def push_matrix(matrix: MapMatrix) -> MapMatrix:
    with open('matrix.json', 'w') as f:
        json.dump(matrix.dict(), f)
    return matrix


@app.get("/matrix/{snapshot}")
async def get_matrix(snapshot: str) -> ResponseMatrix:

    response_matrix = await simulator.get_map_matrix(snapshot=snapshot)
    return response_matrix


@app.get("/")
async def root():
    return {"message": "Hello World"}

