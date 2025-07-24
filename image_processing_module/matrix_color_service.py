from collections import Counter
from dataclasses import dataclass
from typing import List, Tuple
from numpy.typing import NDArray
import numpy as np

from color_utils import ColorUtils

@dataclass
class ParSetColor:
    set: set[Tuple[int,int]]
    color: np.ndarray

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
    
    def expansion(self, n = 500, delta_threshold = 60) -> NDArray[np.float64]:
        return_matrix: NDArray[np.float64] = np.zeros_like(self.matrix)
        visited_matrix: np.ndarray = np.full((self.height, self.width), False, dtype=bool)
        index_set_matrix: np.ndarray = np.full((self.height, self.width), -1, dtype=int)
        #Lista de pares conjunto de colores y color promedio.
        color_set_list: List[ParSetColor] = []

        #Caso base
        par: ParSetColor = ParSetColor({(0,0)},self.matrix[0,0])
        color_set_list.append(par)
        visited_matrix[0,0] = True
        index_set_matrix[0,0] = 0
        self.__expand((0,0),(1,0),visited_matrix,index_set_matrix,color_set_list,delta_threshold)
        self.__expand((0,0),(0,1),visited_matrix,index_set_matrix,color_set_list,delta_threshold)

        color_set_list.sort(key=lambda x: len(x.set), reverse=True)
        #Obtenemos los primeros n conjuntos y restantes
        first_sets = color_set_list[:n]
        last_sets = color_set_list[n:]

        available_colors = []
        #Transforma cada color de los primeros n conjuntos al average correspondiente de su conjunto
        for par in first_sets:
            for tuple in par.set:
                return_matrix[tuple[0],tuple[1]] = par.color
            available_colors.append(par.color)

        #Mejorar para que se haga expansion sobre este conjunto de colores
        #Transforma cada color de los ultimos conjuntos en uno de los average de los primeros n conjuntos
        for par in first_sets:
            for tuple in par.set:
                print("Transforming")
                return_matrix[tuple[0],tuple[1]] = min(available_colors, key=lambda color: ColorUtils.delta_ciede2000(self.matrix[tuple[0],tuple[1]], color))

        return return_matrix

    def __expand(self, prev: Tuple[int,int], p: Tuple[int,int], visited_matrix: np.ndarray, index_set_matrix: np.ndarray, color_set_list: List[ParSetColor], delta_threshold = 20):
        i:int = p[0]
        j:int = p[1]
        if ( i<self.height and j<self.width ):
            prev_set_index:int = index_set_matrix[prev[0],prev[1]]
            prev_par_set_color = color_set_list[prev_set_index]
            average_color_prev_set = prev_par_set_color.color
            prev_delta: float = ColorUtils.delta_ciede2000(average_color_prev_set, self.matrix[i,j])
            print(prev_delta)
            #Mejor delta, hay que insertar en el conjunto anterior

            if (not visited_matrix[i,j]):
                #Primera vez visitado y el delta cumple
                if(prev_delta<delta_threshold):  
                    index_set_matrix[i,j] = prev_set_index
                    self.__add_color_to_set(prev_par_set_color,(i,j))
                #Primera vez visitado y el delta no cumple
                else:
                    color_set_list.append(ParSetColor({(i,j)},self.matrix[i,j]))
                    index_set_matrix[i,j] = len(color_set_list) - 1

                visited_matrix[i,j] = True
                self.__expand((i,j),(i+1,j),visited_matrix,index_set_matrix,color_set_list,delta_threshold)
                self.__expand((i,j),(i,j+1),visited_matrix,index_set_matrix,color_set_list,delta_threshold)

            else:
                set_index = index_set_matrix[i,j]
                par_set_color: ParSetColor = color_set_list[set_index]
                actual_delta = ColorUtils.delta_ciede2000(par_set_color.color, self.matrix[i,j])
                #Habia sido visitado y el delta cumple. Si el color actual estaba solo en un conjunto y el delta cumple se mueve de conjunto.
                if(prev_delta<delta_threshold and (len(par_set_color.set) == 1 or prev_delta < actual_delta)):
                    self.__delete_color_from_set(par_set_color, (i,j))

                    index_set_matrix[i,j] = prev_set_index
                    self.__add_color_to_set(prev_par_set_color,(i,j))
                    self.__expand((i,j),(i+1,j),visited_matrix,index_set_matrix,color_set_list,delta_threshold)
                    self.__expand((i,j),(i,j+1),visited_matrix,index_set_matrix,color_set_list,delta_threshold)

    def __add_color_to_set(self, par: ParSetColor, p: Tuple[int,int]):
        new_color = self.matrix[p[0],p[1]]
        par.color = (par.color * len(par.set) + new_color) / (len(par.set) + 1)
        par.set.add(p)
    
    def __delete_color_from_set(self, par: ParSetColor, p: Tuple[int,int]):
        old_color = self.matrix[p[0],p[1]]
        if len(par.set) > 1:
            par.color = (par.color * len(par.set) - old_color) / (len(par.set) - 1)
        par.set.remove(p)
