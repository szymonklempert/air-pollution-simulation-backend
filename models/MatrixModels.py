from pydantic import BaseModel
from typing import List, Optional


class Cell(BaseModel):
    temp: float
    pressure: Optional[int]
    pm2_5: float
    pm10: float
    wind_speed: Optional[float]
    wind_direction: Optional[int]

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