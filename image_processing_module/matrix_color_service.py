from collections import Counter
from numpy.typing import NDArray
import numpy as np

class MatrixColorService:
    matrix: NDArray[np.float64]

    # Contadores para cada color
    __color_counters = {
        "red": Counter(),
        "black": Counter(),
        "white": Counter(),
        "blue": Counter(),
        "green": Counter(),
        "yellow": Counter(),
        "grey": Counter(),
        "brown": Counter(),
        "other": Counter()
    }

    def __init__(self, matrix: NDArray[np.float64]):
        height, width, lab = matrix.shape
        self.matrix = matrix
        for i in range(height):
            for j in range(width):
                self.save_color(matrix[i][j])

    def save_color(self, lab_color: np.ndarray):
        if lab_color.shape != (3,):
            raise ValueError("lab_color debe ser un array de forma (3,)")
        
        L, a, b = lab_color.astype(int)
        self.__color_counters[self.color_set(lab_color)][(L, a, b)] += 1

    def color_set(self, lab_color: np.ndarray) -> str:
        if lab_color.shape != (3,):
            raise ValueError("lab_color debe ser un array de forma (3,)")

        L, a, b = lab_color.astype(int)
        color = "other"

        if (L < 55 and a > 45 and b > -30) or (a > 100 and b > -25) or (L > 45 and a > 105) or (L < 15 and a > 5 and b > 80) or  (L < 35 and a > 30 and b > -25) :
            color = "red"
        elif L <= 16 and -5 < a < 5 and -5 < b < 5:
            color = "black"
        elif L >= 83 and -2 < a < 2 and -2 < b < 2:
            color = "white"
        elif 16 < L < 83 and -3 < a < 3 and -3 < b < 3:
            color = "grey"
        elif 5 < L < 60 and a < -18 and b < -18:
            color = "blue"
        elif 5 < L < 60 and a < -40 and L > 0:
            color = "green"
        elif L > 67 and -5 < a < 5 and b > 67:
            color = "yellow"
        elif 15 < L < 60 and -5 < a < 15 and b > 10:
            color = "brown"

        return color

    def __get_most_common_color(self, color: str) -> tuple[int, int, int]:
        if color not in self.__color_counters:
            raise ValueError(f"Color desconocido: {color}")
        
        counter = self.__color_counters[color]
        most_common = counter.most_common(1)[0][0] 
        return most_common
    
    def paint_main_colors(self) -> NDArray[np.float64]:
        height, width, lab = self.matrix.shape
        for i in range(height):
            for j in range(width):
                self.matrix[i,j] = self.__get_most_common_color(self.color_set(self.matrix[i,j]))
        return self.matrix