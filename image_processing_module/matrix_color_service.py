from collections import Counter, deque
from dataclasses import dataclass
from typing import List, Tuple
from numpy.typing import NDArray
import numpy as np

from color_utils import ColorUtils

@dataclass
class ParSetColor:
    set: set[Tuple[int,int]]
    color: np.ndarray

@dataclass
class ConjuntoConectado:
    set: set[Tuple[int,int]]
    a: Tuple[int,int]
    b: Tuple[int,int]

class MatrixColorService:
    matrix: NDArray[np.float64]
    height: int
    width: int
    c1 = 1
    c2 = 1

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

    def expansion_bfs(self, n = 12, delta_threshold = 12) -> NDArray[np.float64]:
        visited_matrix: np.ndarray = np.full((self.height, self.width), False, dtype=bool)
        index_set_matrix: np.ndarray = np.full((self.height, self.width), -1, dtype=int)
        #Lista de pares conjunto de colores y color promedio.
        color_set_list: List[ParSetColor] = []
        self.__expand_bfs(visited_matrix,index_set_matrix,color_set_list,delta_threshold)
        color_set_list.sort(key=lambda x: len(x.set), reverse=True)
        #Obtenemos los primeros n conjuntos y restantes
        print(len(color_set_list)," Sets")
        first_sets = color_set_list[:n]
        last_sets = color_set_list[n:]

        available_colors = []
        #Transforma cada color de los primeros n conjuntos al average correspondiente de su conjunto
        for par in first_sets:
            for tuple in par.set:
                self.matrix[tuple[0],tuple[1]] = par.color
            available_colors.append(par.color)

        '''
        p_queue = deque(point for par in last_sets for point in par.set)
        self.__join_neighbors(p_queue)
        '''
        #Mejorar para que se haga expansion sobre este conjunto de colores
        #Transforma cada color de los ultimos conjuntos en uno de los average de los primeros n conjuntos
        for par in last_sets:
            for tuple in par.set:
                self.matrix[tuple[0],tuple[1]] = min(available_colors, key=lambda color: ColorUtils.delta_ciede2000(self.matrix[tuple[0],tuple[1]], color))
        
        return self.matrix

    def __expand_bfs(self, visited_matrix: np.ndarray, index_set_matrix: np.ndarray, color_set_list: List[ParSetColor], delta_threshold = 20):
        i = 0
        j = 0
        par: ParSetColor = ParSetColor({(i,j)},self.matrix[i,j])
        color_set_list.append(par)
        visited_matrix[i,j] = True
        index_set_matrix[i,j] = 0
        queue = deque()
        queue.append(((i,j),(i+1, j)))
        queue.append(((i,j),(i, j+1)))
        while queue:
            prev, p = queue.popleft()
            i,j = p
            neighbors = [((i,j),(i+1, j)), ((i,j),(i,j+1))]
            if ( 0 <= i < self.height and 0 <= j < self.width ):
                prev_set_index:int = index_set_matrix[prev[0],prev[1]]
                prev_par_set_color = color_set_list[prev_set_index]
                average_color_prev_set = prev_par_set_color.color
                prev_delta: float = ColorUtils.delta_ciede2000(average_color_prev_set, self.matrix[i,j])
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
                    for n in neighbors:
                        queue.append(n)
                else:
                    set_index = index_set_matrix[i,j]
                    par_set_color: ParSetColor = color_set_list[set_index]
                    #Habia sido visitado y el delta cumple. Prioriza el conjunto
                    if(prev_delta<delta_threshold and (len(par_set_color.set) < len(prev_par_set_color.set))):
                        self.__delete_color_from_set(par_set_color, (i,j))
                        index_set_matrix[i,j] = prev_set_index
                        self.__add_color_to_set(prev_par_set_color,(i,j))
                        for n in neighbors:
                            queue.append(n)

    def __add_color_to_set(self, par: ParSetColor, p: Tuple[int,int]):
        new_color = self.matrix[p[0],p[1]]
        par.color = (par.color * len(par.set) + new_color) / (len(par.set) + 1)
        par.set.add(p)
    
    def __delete_color_from_set(self, par: ParSetColor, p: Tuple[int,int]):
        old_color = self.matrix[p[0],p[1]]
        if len(par.set) > 1:
            par.color = (par.color * len(par.set) - old_color) / (len(par.set) - 1)
        par.set.remove(p)

    def __join_neighbors(self, p_queue: deque[Tuple[int,int]]):
        while p_queue:
            p = p_queue.popleft()
            i,j = p
            neighbors = [(i-1,j),(i+1,j),(i,j-1),(i,j+1)]
            candidates = [
                (ni, nj) for ni, nj in neighbors
                if 0 <= ni < self.height and 0 <= nj < self.width and (ni, nj) not in p_queue
            ]
            if candidates:
                target_color = self.matrix[i, j]
                best_color = min(
                    (self.matrix[ci, cj] for ci, cj in candidates),
                    key=lambda color: ColorUtils.delta_ciede2000(target_color, color)
                )
                self.matrix[i, j] = best_color
            else:
                p_queue.append(p)

    def delta_matrix(self, threshold = 15) -> NDArray[np.float64]:
        delta_matrix: np.ndarray = np.full((self.height, self.width), 0, dtype=float)
        p_set:set[Tuple[int,int]] = set()
        for i in range(self.height):
            for j in range(self.width):
                delta = self.__delta_vecinos((i,j))
                if delta>threshold:
                    delta_matrix[i,j] = 100
                    p_set.add((i,j))
        self.matrix[:, :, 0] =  delta_matrix
        self.matrix[:, :, 1] = 0                             
        self.matrix[:, :, 2] = 0
        p_set = self.__eliminar_cruzados(p_set)
        set_copy = p_set.copy()
        connected_sets = self.__conjuntos_conectados(set_copy)
        print("Connected sets: ",connected_sets)
        for i in range(self.height):
            for j in range(self.width):
                if (i,j) not in p_set:
                    self.matrix[i, j, 0] = 0
        return self.matrix 
        
    def __delta_vecinos(self, p: Tuple[int, int]) -> np.float64:
        i, j = p
        max_delta = 0.0

        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ni, nj = i + di, j + dj
            if 0 <= ni < self.height and 0 <= nj < self.width:
                delta = ColorUtils.delta_ciede2000(self.matrix[i, j], self.matrix[ni, nj])
                max_delta = max(max_delta, delta)

        return max_delta

    def __eliminar_cruzados(self, points: set[tuple[int, int]]) -> set[tuple[int, int]]:
        # Crear estructuras para acceso rápido
        from collections import defaultdict

        col_map = defaultdict(set)  # j -> set of i
        row_map = defaultdict(set)  # i -> set of j

        for i, j in points:
            col_map[j].add(i)
            row_map[i].add(j)

        result = set()
        for i, j in points:
            arriba = any(ii < i for ii in col_map[j])
            abajo  = any(ii > i for ii in col_map[j])
            izq    = any(jj < j for jj in row_map[i])
            der    = any(jj > j for jj in row_map[i])

            if not (arriba and abajo and izq and der):
                result.add((i, j))

        return result
    
    def __conjuntos_conectados(self, p_set :set[Tuple[int,int]]) -> List[ConjuntoConectado]:
        toReturn: List[ConjuntoConectado] = []
        while(len(p_set)>0):
            p = next(iter(p_set))
            s:set = {p}
            conected_set:ConjuntoConectado = ConjuntoConectado(s,p,p)
            adyacentes = self.__obtener_adyacentes(p,p_set)
            if len(adyacentes)>0:
                conectados = self.__conectados(p,adyacentes[0],p_set)
                conected_set.a = conectados[-1]
                s.update(conectados)
            if len(adyacentes)>1:
                conectados = self.__conectados(p,adyacentes[1],p_set)
                conected_set.b = conectados[-1]
                s.update(conectados)
            toReturn.append(conected_set)
            p_set -= s
        return toReturn

    def __conectados(self, prev: Tuple[int, int], p: Tuple[int, int], p_set: set[Tuple[int, int]]) -> List[Tuple[int, int]]:
        p_set.discard(p)
        adyacentes = self.__obtener_adyacentes(p, p_set)
        if len(adyacentes) > 1:
            next_p = adyacentes[0] if adyacentes[0] != prev else adyacentes[1]
            return [p] + self.__conectados(p, next_p, p_set)
        else:
            return [p]

    def __obtener_adyacentes(self, p: Tuple[int,int], c: set[Tuple[int,int]]) -> List[Tuple[int,int]]:
        i, j = p
        vecinos_relativos = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),          (0, 1),
            (1, -1),  (1, 0), (1, 1)
        ]
        
        return [(i + di, j + dj) for di, dj in vecinos_relativos if (i + di, j + dj) in c]
