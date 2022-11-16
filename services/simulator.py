import json
from models.MatrixModels import MapMatrix, Cell, ResponseMatrix
from typing import Dict, List
import numpy as np

WINDOW_SIZE = 3

async def get_map_matrix(snapshot: str) -> ResponseMatrix:
    with open('matrix.json') as f:
        json_matrix = json.load(f)
        print(type(json_matrix))
    
    sim_dict: Dict[str, MapMatrix] = await get_simulation(matrix=MapMatrix.parse_obj(json_matrix))
    # return sim_dict[snapshot]
    res_matrix: MapMatrix = sim_dict[snapshot]
    return ResponseMatrix(x=res_matrix.x, y=res_matrix.y, k=res_matrix.k, snapshot=snapshot, matrix=res_matrix.matrix)


async def get_simulation_dict() -> Dict[str, MapMatrix]:
    pass


async def get_simulation(matrix: MapMatrix) -> Dict[str, MapMatrix]:
    np_matrix = parse_matrix(matrix.matrix, matrix.x, matrix.y)
    simulation_dict = {}


    previous_np_matrix = simulate(np_matrix, k=matrix.k)

    for i in range(100):
        curr_np_matrix = simulate(previous_np_matrix, k=matrix.k)

        cell_list = get_cell_list(curr_np_matrix)
        simulation_dict[str(i)] = MapMatrix(x=matrix.x, y=matrix.y, k=matrix.k, matrix=cell_list)

        previous_np_matrix = curr_np_matrix
    return simulation_dict

def simulate(matrix, k):
    D = k * 0.1

    # res_matrix = np.empty(matrix.shape, dtype=object)\
    res_matrix = [[Cell(temp=0, pressure=0, pm2_5=0, pm10=0) for y in range(matrix.shape[1])] for x in range(matrix.shape[0])]

    for x in range(1, matrix.shape[0]):
        for y in range(1, matrix.shape[1]):

            cur_window: List[List[Cell]] = matrix[x-1:x+2, y-1:y+2]
            # cur_window = cur_window.astype(np.int32)

            # print(x, y)
            # print(lenaNoiseRead[x-1:x+2, y-1:y+2])

            # for x_window in range(cur_window.shape[0]):
            #     for y_window in range(cur_window.shape[1]):
            current_cell: Cell = cur_window[1][1]
            res_pm10 = D * (
                (current_cell.pm10 - cur_window[0][1].pm10) + 
                (current_cell.pm10 - cur_window[1][0].pm10) + 
                (current_cell.pm10 - cur_window[1][2].pm10) + 
                (current_cell.pm10 - cur_window[2][1].pm10) 
            )
            res_pm10 += 1 / np.sqrt(2) * D * (
                (current_cell.pm10 - cur_window[0][0].pm10) + 
                (current_cell.pm10 - cur_window[0][2].pm10) + 
                (current_cell.pm10 - cur_window[2][0].pm10) + 
                (current_cell.pm10 - cur_window[2][2].pm10) 
            )
            
            res_matrix[x][y].pm10 = res_pm10 

    return res_matrix

                    


# def simulate(cell: Cell) -> Cell:
#     # temporary simple simulation
#     return Cell(temp=cell.temp-0.5, pm2_5=f(cell.temp).real, pm10=f(cell.temp).real)


def f(t: float) -> complex:
    # -(29/30)^(x+40) + 1
    p = -29/30
    return (p**(t+40)) + 1
    

def get_cell_list(matrix: np.ndarray) -> List[Cell]:
    # TODO: iterate in numpy way
    res = list()
    for x in matrix:
        for y in x:
            res.append(y)
    return res


def parse_matrix(cells_list: List[Cell], n_cols: int, n_rows: int) -> np.ndarray:
    # TODO: parse it in numpy way
    matrix = list()
    for r in range(n_rows):
        row = list()
        for c in range(n_cols):
            row.append(cells_list[n_cols * r + c])
        row = np.array(row)
        matrix.append(row)
    
    return np.array(matrix)

            

