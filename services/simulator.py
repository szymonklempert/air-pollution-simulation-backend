import json
from models.MatrixModels import MapMatrix, Cell, ResponseMatrix
from typing import Dict, List, Tuple
import numpy as np
import random


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
    res_matrix: np.ndarray(dtype=Cell) = parse_matrix(x_size, y_size)

    for x in range(x_size):
        for y in range(y_size):
            res_pm10 = 0
            res_pm2_5 = 0

            current_cell = matrix[x][y]

            if x == 0 and y == 0:
                res_pm2_5, res_pm10 = calculate_pm_diffusion(
                    current_cell=current_cell,
                    cells=[matrix[x][y + 1], matrix[x + 1][y]],
                    diagonal_cells=[matrix[x + 1][y + 1]],
                    k=k,
                )

            elif x == 0 and y == y_size - 1:
                res_pm2_5, res_pm10 = calculate_pm_diffusion(
                    current_cell=current_cell,
                    cells=[matrix[x][y - 1], matrix[x + 1][y]],
                    diagonal_cells=[matrix[x + 1][y - 1]],
                    k=k,
                )

            elif x == x_size - 1 and y == y_size - 1:
                res_pm2_5, res_pm10 = calculate_pm_diffusion(
                    current_cell=current_cell,
                    cells=[matrix[x][y - 1], matrix[x - 1][y]],
                    diagonal_cells=[matrix[x - 1][y - 1]],
                    k=k,
                )

            elif x == x_size - 1 and y == 0:
                res_pm2_5, res_pm10 = calculate_pm_diffusion(
                    current_cell=current_cell,
                    cells=[matrix[x][y + 1], matrix[x - 1][y]],
                    diagonal_cells=[matrix[x - 1][y + 1]],
                    k=k,
                )

            elif x == 0:
                res_pm2_5, res_pm10 = calculate_pm_diffusion(
                    current_cell=current_cell,
                    cells=[matrix[x][y + 1], matrix[x][y - 1], matrix[x + 1][y]],
                    diagonal_cells=[matrix[x + 1][y + 1], matrix[x + 1][y - 1]],
                    k=k,
                )

            elif y == 0:
                res_pm2_5, res_pm10 = calculate_pm_diffusion(
                    current_cell=current_cell,
                    cells=[matrix[x + 1][y], matrix[x - 1][y], matrix[x][y + 1]],
                    diagonal_cells=[matrix[x + 1][y + 1], matrix[x - 1][y + 1]],
                    k=k,
                )

            elif x == x_size - 1:
                res_pm2_5, res_pm10 = calculate_pm_diffusion(
                    current_cell=current_cell,
                    cells=[matrix[x][y + 1], matrix[x][y - 1], matrix[x - 1][y]],
                    diagonal_cells=[matrix[x - 1][y + 1], matrix[x - 1][y - 1]],
                    k=k,
                )

            elif y == y_size - 1:
                res_pm2_5, res_pm10 = calculate_pm_diffusion(
                    current_cell=current_cell,
                    cells=[matrix[x + 1][y], matrix[x - 1][y], matrix[x][y - 1]],
                    diagonal_cells=[matrix[x + 1][y - 1], matrix[x - 1][y - 1]],
                    k=k,
                )

            else:
                cur_window: List[List[Cell]] = matrix[x - 1 : x + 2, y - 1 : y + 2]
                current_window_cell: Cell = cur_window[1][1]

                res_pm2_5, res_pm10 = calculate_pm_diffusion(
                    current_cell=current_window_cell,
                    cells=[
                        cur_window[0][1],
                        cur_window[1][0],
                        cur_window[1][2],
                        cur_window[2][1],
                    ],
                    diagonal_cells=[
                        cur_window[0][0],
                        cur_window[0][2],
                        cur_window[2][0],
                        cur_window[2][2],
                    ],
                    k=k,
                )

            pm2_5_clapeyron, pm10_clapeyron = calculate_pm_clapeyron(current_cell)
            res_pm2_5 += pm2_5_clapeyron
            res_pm10 += pm10_clapeyron

            res_matrix[x][y].pm2_5 = res_pm2_5
            res_matrix[x][y].pm10 = res_pm10

    return matrix - res_matrix


def calculate_pm_clapeyron(current_cell: Cell) -> Tuple[float, float]:
    """
    Calculate the pm based on The Clapeyron Equation
    """

    temp_delta = random.uniform(-1.0, 1.0)
    pressure_delta = random.randint(-10, 10)

    t0 = current_cell.temp
    t1 = current_cell.temp + temp_delta

    if current_cell.pressure:
        p0 = current_cell.pressure
        p1 = current_cell.pressure + pressure_delta
    else:
        p0 = 1012
        p1 = 1012 + pressure_delta

    pm2_5 = current_cell.pm2_5 * ((t0 / t1) * (p1 / p0))
    pm10 = current_cell.pm10 * ((t0 / t1) * (p1 / p0))

    # print("pm10", current_cell.pm10)
    # print("t", t1 / t0)
    # print("p", p1 / p0)
    # print("res", ((t0 / t1) * (p1 / p0)))
    # print(pm2_5, pm10)
    return pm2_5 - current_cell.pm2_5, pm10 - current_cell.pm10


def calculate_pm_diffusion(
    current_cell: Cell, cells: List[Cell], diagonal_cells: List[Cell], k: int
) -> Tuple[float, float]:
    """
    Calculate the pm based on The Diffusion Equation
    """

    pm2_5 = calculate_pm2_5(current_cell, cells, diagonal_cells, k)
    pm10 = calculate_pm10(current_cell, cells, diagonal_cells, k)

    return pm2_5, pm10


def calculate_pm2_5(
    current_cell: Cell, cells: List[Cell], diagonal_cells: List[Cell], k: int
):
    D = k * 0.245

    new_pm2_5 = 0

    for cell in cells:
        new_pm2_5 += current_cell.pm2_5 - cell.pm2_5

    new_pm2_5 *= D

    for d_cell in diagonal_cells:
        new_pm2_5 += current_cell.pm2_5 - d_cell.pm2_5

    new_pm2_5 *= 1 / np.sqrt(2)
    new_pm2_5 *= D

    return new_pm2_5


def calculate_pm10(
    current_cell: Cell, cells: List[Cell], diagonal_cells: List[Cell], k: int
):
    D = k * 0.245

    new_pm10 = 0

    for cell in cells:
        new_pm10 += current_cell.pm10 - cell.pm10

    new_pm10 *= D

    for d_cell in diagonal_cells:
        new_pm10 += current_cell.pm10 - d_cell.pm10

    new_pm10 *= 1 / np.sqrt(2)
    new_pm10 *= D

    return new_pm10


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
