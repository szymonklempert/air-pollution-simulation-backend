import json
from models.MatrixModels import MapMatrix, Cell, ResponseMatrix
from typing import Dict, List
import numpy as np

WINDOW_SIZE = 3


async def get_response_matrix(snapshot: str) -> ResponseMatrix:
    with open("matrix.json") as f:
        json_matrix = json.load(f)

    sim_dict: Dict[str, MapMatrix] = await get_simulation(
        matrix=MapMatrix.parse_obj(json_matrix)
    )
    # return sim_dict[snapshot]
    res_matrix: MapMatrix = sim_dict[snapshot]
    return ResponseMatrix(
        x=res_matrix.x,
        y=res_matrix.y,
        k=res_matrix.k,
        snapshot=snapshot,
        matrix=res_matrix.matrix,
    )


async def get_simulation(matrix: MapMatrix) -> Dict[str, MapMatrix]:
    np_matrix = parse_matrix(matrix.x, matrix.y, matrix.matrix)
    simulation_dict = {}

    previous_np_matrix = simulate(
        np_matrix, k=matrix.k, x_size=matrix.x, y_size=matrix.y
    )

    for i in range(100):
        curr_np_matrix = simulate(
            previous_np_matrix, k=matrix.k, x_size=matrix.x, y_size=matrix.y
        )

        cell_list = get_cell_list(curr_np_matrix)
        simulation_dict[str(i)] = MapMatrix(
            x=matrix.x, y=matrix.y, k=matrix.k, matrix=cell_list
        )

        previous_np_matrix = curr_np_matrix
    return simulation_dict


def simulate(matrix: np.ndarray, k: float, x_size: int, y_size: int):
    D = k * 0.245

    res_matrix = parse_matrix(x_size, y_size)

    for x in range(x_size - 1):
        for y in range(y_size - 1):
            res_pm10 = 0
            res_pm2_5 = 0

            if x == 0 and y == 0:
                res_pm10 += D * (
                    (matrix[x][y].pm10 - matrix[x][y + 1].pm10)
                    + (matrix[x][y].pm10 - matrix[x + 1][y].pm10)
                )
                res_pm10 += (
                    1 / np.sqrt(2) * D * (matrix[x][y].pm10 - matrix[x + 1][y + 1].pm10)
                )

                res_pm2_5 += D * (
                    (matrix[x][y].pm2_5 - matrix[x][y + 1].pm2_5)
                    + (matrix[x][y].pm2_5 - matrix[x + 1][y].pm2_5)
                )
                res_pm2_5 += (
                    1
                    / np.sqrt(2)
                    * D
                    * (matrix[x][y].pm2_5 - matrix[x + 1][y + 1].pm2_5)
                )

            elif x == 0:
                res_pm10 += D * (
                    (matrix[x][y].pm10 - matrix[x][y + 1].pm10)
                    + (matrix[x][y].pm10 - matrix[x][y - 1].pm10)
                    + (matrix[x][y].pm10 - matrix[x + 1][y].pm10)
                )
                res_pm10 += (
                    1
                    / np.sqrt(2)
                    * D
                    * (
                        (matrix[x][y].pm10 - matrix[x + 1][y + 1].pm10)
                        + (matrix[x][y].pm10 - matrix[x + 1][y - 1].pm10)
                    )
                )

                res_pm2_5 += D * (
                    (matrix[x][y].pm2_5 - matrix[x][y + 1].pm2_5)
                    + (matrix[x][y].pm2_5 - matrix[x][y - 1].pm2_5)
                    + (matrix[x][y].pm2_5 - matrix[x + 1][y].pm2_5)
                )
                res_pm2_5 += (
                    1
                    / np.sqrt(2)
                    * D
                    * (
                        (matrix[x][y].pm2_5 - matrix[x + 1][y + 1].pm2_5)
                        + (matrix[x][y].pm2_5 - matrix[x + 1][y - 1].pm2_5)
                    )
                )

            elif y == 0:
                res_pm10 += D * (
                    (matrix[x][y].pm10 - matrix[x + 1][y].pm10)
                    + (matrix[x][y].pm10 - matrix[x - 1][y].pm10)
                    + (matrix[x][y].pm10 - matrix[x][y + 1].pm10)
                )
                res_pm10 += (
                    1
                    / np.sqrt(2)
                    * D
                    * (
                        (matrix[x][y].pm10 - matrix[x + 1][y + 1].pm10)
                        + (matrix[x][y].pm10 - matrix[x - 1][y + 1].pm10)
                    )
                )

                res_pm2_5 += D * (
                    (matrix[x][y].pm2_5 - matrix[x + 1][y].pm2_5)
                    + (matrix[x][y].pm2_5 - matrix[x - 1][y].pm2_5)
                    + (matrix[x][y].pm2_5 - matrix[x][y + 1].pm2_5)
                )
                res_pm2_5 += (
                    1
                    / np.sqrt(2)
                    * D
                    * (
                        (matrix[x][y].pm2_5 - matrix[x + 1][y + 1].pm2_5)
                        + (matrix[x][y].pm2_5 - matrix[x - 1][y + 1].pm2_5)
                    )
                )



            else:
                cur_window: List[List[Cell]] = matrix[x - 1 : x + 2, y - 1 : y + 2]
                current_cell: Cell = cur_window[1][1]

                res_pm10 += D * (
                    (current_cell.pm10 - cur_window[0][1].pm10)
                    + (current_cell.pm10 - cur_window[1][0].pm10)
                    + (current_cell.pm10 - cur_window[1][2].pm10)
                    + (current_cell.pm10 - cur_window[2][1].pm10)
                )
                res_pm10 += (
                    1
                    / np.sqrt(2)
                    * D
                    * (
                        (current_cell.pm10 - cur_window[0][0].pm10)
                        + (current_cell.pm10 - cur_window[0][2].pm10)
                        + (current_cell.pm10 - cur_window[2][0].pm10)
                        + (current_cell.pm10 - cur_window[2][2].pm10)
                    )
                )

                res_pm2_5 += D * (
                    (current_cell.pm2_5 - cur_window[0][1].pm2_5)
                    + (current_cell.pm2_5 - cur_window[1][0].pm2_5)
                    + (current_cell.pm2_5 - cur_window[1][2].pm2_5)
                    + (current_cell.pm2_5 - cur_window[2][1].pm2_5)
                )
                res_pm2_5 += (
                    1
                    / np.sqrt(2)
                    * D
                    * (
                        (current_cell.pm2_5 - cur_window[0][0].pm2_5)
                        + (current_cell.pm2_5 - cur_window[0][2].pm2_5)
                        + (current_cell.pm2_5 - cur_window[2][0].pm2_5)
                        + (current_cell.pm2_5 - cur_window[2][2].pm2_5)
                    )
                )

            res_matrix[x][y].pm10 = res_pm10
            res_matrix[x][y].pm10 = res_pm10

    return matrix - res_matrix


def get_cell_list(matrix: np.ndarray) -> List[Cell]:
    # TODO: iterate in numpy way
    res = list()
    for x in matrix:
        for y in x:
            res.append(y)
    return res


def parse_matrix(n_cols: int, n_rows: int, cells_list: List[Cell] = None) -> np.ndarray:
    if not cells_list:
        cells_list = [
            Cell(temp=0, pressure=0, pm2_5=0, pm10=0) for _ in range(n_cols * n_rows)
        ]
    # TODO: parse it in numpy way
    matrix = list()
    for r in range(n_rows):
        row = list()
        for c in range(n_cols):
            row.append(cells_list[n_cols * r + c])
        row = np.array(row)
        matrix.append(row)

    return np.array(matrix)
