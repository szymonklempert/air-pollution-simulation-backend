from pydantic import BaseModel
from typing import List, Optional


class Cell(BaseModel):
    temp: float
    pressure: Optional[int]
    pm2_5: float
    pm10: float
    wind_speed: Optional[float]
    wind_direction: Optional[int]

    def __sub__(self, other):
        # print(other.pm10)
        return Cell(
            temp=self.temp,
            pressure=self.pressure,
            pm2_5=self.pm2_5 - other.pm2_5,
            pm10=self.pm10 - other.pm10,
            wind_speed=self.wind_speed,
            wind_direction=self.wind_direction,
        )


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
