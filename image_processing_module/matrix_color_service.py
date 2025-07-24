from collections import Counter
from typing import List, Tuple
from numpy.typing import NDArray
import numpy as np

class MatrixColorService:
    matrix: NDArray[np.float64]

    # Contador para todos los colores
    __color_counter =  Counter()
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
        self.__color_counter[(L, a, b)] += 1

    def color_set(self, lab_color: np.ndarray) -> str:
        if lab_color.shape != (3,):
            raise ValueError("lab_color debe ser un array de forma (3,)")

        L, a, b = lab_color.astype(int)
        color = "other"

        if ((L < 55 and a > 45 and b > -30) or (a > 100 and b > -70) or (L > 45 and a > 105) or (L < 10 and a > -30 and b > 105) or
            (L < 15 and a > 5 and b > 80) or  (L < 35 and a > 30 and b > -25) and (a > 25 and b > 0) or  (L < 20 and a > 15 and b > 5)):
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
        height, width, _ = self.matrix.shape
        return_matrix: NDArray[np.float64] = np.zeros_like(self.matrix)  # Inicializa la matriz de salida

        for i in range(height):
            for j in range(width):
                color = self.matrix[i, j]
                color_set = self.color_set(color)
                main_color = self.__get_most_common_color(color_set)
                return_matrix[i, j] = main_color

        return return_matrix
    
    def expansion(self):
        height, width, _ = self.matrix.shape
        visited_matrix: np.ndarray = np.full((height, width), False, dtype=bool)
        id_set_matrix: np.ndarray = np.full((height, width), -1, dtype=int)
        #Lista de pares conjunto de colores y color promedio.
        color_set_list: List[Tuple[set[np.ndarray],np.ndarray]]

    def __add_color_to_set(tuple: Tuple[set[np.ndarray],np.ndarray], new_color:np.ndarray):
        tuple[1] =  (tuple[1] * len(tuple[0]) + new_color) / (len(tuple[0]) + 1)
        tuple[0].add(new_color)
    
    def __delete_color_from_set(tuple: Tuple[set[np.ndarray],np.ndarray], old_color:np.ndarray):
        if len(tuple[0]) > 1:
            tuple[1] = (tuple[1] * len(tuple[0]) - old_color) / (len(tuple[0]) - 1)
        tuple[0].remove(old_color)

    '''
    Algoritmo de expansion:
    Estructuras:
    -Matriz de colores
    -Matriz booleana de visitados
    -Lista de tupla (color promedio, conjuntos de colores (i,j))
    -Matriz idConjunto (posicion del conjunto en la lista) | None



    Algoritmo:
    -Se empieza por [0,0]
    -Se expande a [i+1,j] y [i,j+1]. Cuando un nodo es alcanzado por la expansion se marca como visitado
    -Si fue visitado, este no se expande. Pero si analiza el delta del nuevo conjunto a ver si le conviene meterse.
    -El conjunto tiene un color promedio, cuando le llega la expansion a ese nodo consulta el delta con el color promedio, si es menor a cierto valor se incorpora al conjunto
    caso contrario crea su propio conjunto.
    
    
    '''