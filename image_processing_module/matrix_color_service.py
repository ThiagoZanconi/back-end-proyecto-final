from collections import Counter
from typing import List, Tuple
from numpy.typing import NDArray
import numpy as np

from color_utils import ColorUtils

class MatrixColorService:
    matrix: NDArray[np.float64]
    height: int
    width: int

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
        self.height, self.width, lab = matrix.shape
        self.matrix = matrix
        for i in range(self.height):
            for j in range(self.width):
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
        return_matrix: NDArray[np.float64] = np.zeros_like(self.matrix)  # Inicializa la matriz de salida

        for i in range(self.height):
            for j in range(self.width):
                color = self.matrix[i, j]
                color_set = self.color_set(color)
                main_color = self.__get_most_common_color(color_set)
                return_matrix[i, j] = main_color

        return return_matrix
    
    '''
    Algoritmo de expansion:
    Estructuras:
    -Matriz de colores
    -Matriz booleana de visitados
    -Lista de tupla (conjuntos de colores (i,j), color promedio)
    -Matriz indice_conjunto (posicion del conjunto en la lista)

    Algoritmo:
    -Se empieza por [0,0]
    -Se expande a [i+1,j] y [i,j+1]. Cuando un nodo es alcanzado por la expansion se marca como visitado
    -Si fue visitado, este no se expande. Pero si analiza el delta del nuevo conjunto a ver si le conviene meterse. Si le conviene meterse, se expande.
    -El conjunto tiene un color promedio, cuando le llega la expansion a ese nodo consulta el delta con el color promedio, si es menor a cierto valor se incorpora al conjunto
    caso contrario crea su propio conjunto.
    '''
    
    def expansion(self, n = 10, delta_threshold = 20) -> NDArray[np.float64]:
        return_matrix: NDArray[np.float64] = np.zeros_like(self.matrix)
        visited_matrix: np.ndarray = np.full((self.height, self.width), False, dtype=bool)
        index_set_matrix: np.ndarray = np.full((self.height, self.width), -1, dtype=int)
        #Lista de pares conjunto de colores y color promedio.
        color_set_list: List[Tuple[set[Tuple[int,int]],np.ndarray]] = []

        #Caso base
        color_set_list.append(({(0,0)}, self.matrix[0,0]))
        visited_matrix[0,0] = True
        index_set_matrix[0,0] = 0
        self.__expand((0,0),(1,0),visited_matrix,index_set_matrix,color_set_list,delta_threshold)
        self.__expand((0,0),(0,1),visited_matrix,index_set_matrix,color_set_list,delta_threshold)

        color_set_list.sort(key=lambda x: len(x[0]), reverse=True)
        #Obtenemos los primeros n conjuntos y restantes
        first_sets = color_set_list[:n]
        last_sets = color_set_list[n:]

        available_colors = []
        #Transforma cada color de los primeros n conjuntos al average correspondiente de su conjunto
        for tuple_set_average in first_sets:
            for tuple in tuple_set_average[0]:
                return_matrix[tuple[0],tuple[1]] = tuple_set_average[1]
                available_colors.append(tuple_set_average[1])

        #Transforma cada color de los ultimos conjuntos en uno de los average de los primeros n conjuntos
        for tuple_set_average in first_sets:
            for tuple in tuple_set_average[0]:
                return_matrix[tuple[0],tuple[1]] = min(available_colors, key=lambda color: ColorUtils.delta_ciede2000(self.matrix[tuple[0],tuple[1]], color))

        return return_matrix

    def __expand(self, prev: Tuple[int,int], p: Tuple[int,int], visited_matrix: np.ndarray, index_set_matrix: np.ndarray, color_set_list: List[Tuple[set[Tuple[int,int]],np.ndarray]], delta_threshold = 20):
        i:int = p[0]
        j:int = p[1]
        if ( i<self.height and j<self.width ):
            prev_set_index:int = index_set_matrix[prev[0],prev[1]]
            prev_tuple_set_average = color_set_list[prev_set_index]
            average_color_prev_set = color_set_list[prev_set_index][1]
            delta: float = ColorUtils.delta_ciede2000(average_color_prev_set, self.matrix[i,j])
            
            #Mejor delta, hay que insertar en el conjunto anterior
            if(delta<delta_threshold):
                #Este nodo ya fue visitado, hay que eliminarlo del conjunto donde esta
                if (visited_matrix[i,j]):
                    set_index = index_set_matrix[i,j]
                    tuple_set_average = color_set_list[set_index]
                    self.__delete_color_from_set(tuple_set_average, (i,j))
                
                index_set_matrix[i,j] = prev_set_index
                self.__add_color_to_set(prev_tuple_set_average,(i,j))
                visited_matrix[i,j] = True
                self.__expand((i,j),(i+1,j),visited_matrix,index_set_matrix,color_set_list,delta_threshold)
                self.__expand((i,j),(i,j+1),visited_matrix,index_set_matrix,color_set_list,delta_threshold)

            #Si no habia sido visitado, crea su propio conjunto y se expande 
            elif (not visited_matrix[i,j]):
                visited_matrix[i,j] = True

                color_set_list.append(({(i,j)}, self.matrix[i,j]))
                visited_matrix[i,j] = True
                index_set_matrix[i,j] = len(color_set_list) - 1

                self.__expand((i,j),(i+1,j),visited_matrix,index_set_matrix,color_set_list,delta_threshold)
                self.__expand((i,j),(i,j+1),visited_matrix,index_set_matrix,color_set_list,delta_threshold)

    def __add_color_to_set(self, tuple: Tuple[set[Tuple[int,int]],np.ndarray], p: Tuple[int,int]):
        new_color = self.matrix[p[0],p[1]]
        tuple[1] = (tuple[1] * len(tuple[0]) + new_color) / (len(tuple[0]) + 1)
        tuple[0].add(p)
    
    def __delete_color_from_set(self, tuple: Tuple[set[Tuple[int,int]],np.ndarray], p: Tuple[int,int]):
        old_color = self.matrix[p[0],p[1]]
        if len(tuple[0]) > 1:
            tuple[1] = (tuple[1] * len(tuple[0]) - old_color) / (len(tuple[0]) - 1)
        tuple[0].remove(p)
