from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


from models.MatrixModels import MapMatrix, ResponseMatrix
from services import simulator
import json

origins = [
    "http://localhost/",
    "http://localhost:4200/",
]

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=[""],
    allow_headers=[""],
)


@app.post("/matrix/")
async def push_matrix(matrix: MapMatrix) -> MapMatrix:
    with open("matrix.json", "w") as f:
        json.dump(matrix.dict(), f)
    return matrix


@app.get("/matrix/{snapshot}")
async def get_matrix(snapshot: str) -> ResponseMatrix:
    response_matrix = await simulator.get_response_matrix(snapshot=snapshot)
    return response_matrix


@app.get("/")
async def root():
    return {"message": "Hello World"}
