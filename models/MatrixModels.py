from pydantic import BaseModel
from typing import List


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
    snapshot: str
    matrix: List[Cell]