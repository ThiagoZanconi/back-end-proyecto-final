from collections import Counter, deque
from dataclasses import dataclass
import math
from typing import Dict, List, Tuple
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
    matrix_shape: NDArray[np.float64]
    boolean_matrix_shape: NDArray
    background_set: set[Tuple[int,int]]
    shape_set: set[Tuple[int,int]]
    height: int
    width: int

    # Contador para todos los colores
    __color_counter =  Counter()

    def __init__(self, matrix: NDArray[np.float64], delta_threshold = 8):
        self.height, self.width, lab = matrix.shape
        self.matrix = matrix
        self.background_set = set()
        self.shape_set = set()
        self.__shape_matrix(delta_threshold)
        for i in range(self.height):
            for j in range(self.width):
                self.save_color(matrix[i][j])

    def save_color(self, lab_color: np.ndarray):
        if lab_color.shape != (3,):
            raise ValueError("lab_color debe ser un array de forma (3,)")
        L, a, b = lab_color.astype(int)
        self.__color_counter[(L, a, b)] += 1

    def remover_fondo(self) -> NDArray[np.float64]:
        matriz_sin_fondo = self.matrix.copy()
        for i in range(self.height):
            for j in range(self.width):
                if(not self.boolean_matrix_shape[i,j]):
                    matriz_sin_fondo[i,j] = [0,0,0]
        return matriz_sin_fondo
    
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

    def __shape_matrix(self, threshold) -> NDArray[np.float64]:
        self.boolean_matrix_shape = np.zeros((self.height, self.width), dtype=bool)
        self.matrix_shape = np.empty_like(self.matrix)
        for i in range(self.height):
            for j in range(self.width):
                delta = self.__delta_vecinos((i,j))
                if delta>threshold:
                    self.boolean_matrix_shape[i,j] = True
                    self.shape_set.add((i,j))
                else:
                    self.background_set.add((i,j))
        self.__connect_borders()
        self.__fill_gaps()
        connected_sets = self.__conjuntos_conectados()
        self.shape_set = connected_sets[0].set
        self.__tapar_picos_negros()
        #self.__extraer_borde_numpy()
        #self.__conectar_diagonales()
        for i in range(self.height):
            for j in range(self.width):
                if (i,j) in self.shape_set:
                    self.matrix_shape[i, j] = [100,0,0]
                else:
                    self.matrix_shape[i, j] = [0,0,0]
        return self.matrix_shape 
        
    def __delta_vecinos(self, p: Tuple[int, int]) -> np.float64:
        i, j = p
        max_delta = 0.0

        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ni, nj = i + di, j + dj
            if 0 <= ni < self.height and 0 <= nj < self.width:
                delta = ColorUtils.delta_ciede2000(self.matrix[i, j], self.matrix[ni, nj])
                max_delta = max(max_delta, delta)

        return max_delta
    
    def __connect_borders(self):
        top_row_length = 0
        bottom_row_length = 0
        left_column_length = 0
        right_column_length = 0
        for i in range(self.height):
            if(self.boolean_matrix_shape[i,0]):
                if(left_column_length != 0):
                    for n in range(i, i - left_column_length, -1):
                        self.boolean_matrix_shape[n,0] = True
                        self.shape_set.add((n,0))
                left_column_length = 1
            elif left_column_length!=0:
                left_column_length+=1

            if(self.boolean_matrix_shape[i,self.width-1]):
                if(right_column_length != 0):
                    for n in range(i, i - right_column_length, -1):
                        self.boolean_matrix_shape[n,self.width-1] = True
                        self.shape_set.add((n,self.width-1))
                right_column_length = 1
            elif right_column_length!=0:
                right_column_length+=1

        for j in range(self.width):
            if(self.boolean_matrix_shape[0,j]):
                if(top_row_length != 0):
                    for n in range(j, j - top_row_length, -1):
                        self.boolean_matrix_shape[0,n] = True
                        self.shape_set.add((0,n))
                top_row_length = 1
            elif top_row_length!=0:
                top_row_length+=1

            if(self.boolean_matrix_shape[self.height-1,j]):
                if(bottom_row_length != 0):
                    for n in range(j, j - bottom_row_length, -1):
                        self.boolean_matrix_shape[self.height-1,n] = True
                        self.shape_set.add((self.height-1,n))
                bottom_row_length = 1
            elif bottom_row_length!=0:
                bottom_row_length+=1
    
    def __fill_gaps(self ) -> set[Tuple[int,int]]:
        border_set = self.shape_set.copy()
        while(border_set):
            adyacentes_not_in_set: set[Tuple[int,int]] = self.__obtener_adyacentes_not_in_set(border_set)
            border_set = self.__agregar_cruzados(adyacentes_not_in_set)

    def __obtener_adyacentes_not_in_set(self, border_set: set[Tuple[int,int]]):
        toReturn: set[Tuple[int,int]] = set()
        for i,j in iter(border_set):
            vecinos = [(i-1,j-1), (i,j-1), (i-1,j), (i+1,j), (i,j+1), (i-1,j+1), (i+1,j-1), (i+1,j+1)]
            for vi,vj in vecinos:
                if( 0 <= vi < self.height and 0 <= vj < self.width and (vi,vj) not in self.shape_set):
                    toReturn.add((vi,vj))
        return toReturn
    
    def __agregar_cruzados(self, adyacentes_not_in_set: List[Tuple[int,int]]) -> set[tuple[int, int]]:
        # Crear estructuras para acceso rápido
        from collections import defaultdict

        col_map = defaultdict(set)  # j -> set of i
        row_map = defaultdict(set)  # i -> set of j

        for i, j in self.shape_set:
            col_map[j].add(i)
            row_map[i].add(j)

        added = set()
        for i, j in adyacentes_not_in_set:
            arriba = any(ii < i for ii in col_map[j])
            abajo  = any(ii > i for ii in col_map[j])
            izq    = any(jj < j for jj in row_map[i])
            der    = any(jj > j for jj in row_map[i])

            if (arriba and abajo and izq and der):
                self.shape_set.add((i, j))
                self.boolean_matrix_shape[i,j] = True
                added.add((i, j))

        return added

    #Conecta los puntos que solo estan unidos por diagonales, de forma directa
    def __conectar_diagonales(self):
        
        p_queue: deque[Tuple[int, int]] = deque(self.shape_set)
        while(p_queue):
            p = p_queue.popleft()
            i,j = p
            vecinos_diagonales = self.__obtener_adyacentes_diagonales(p)
            for v in vecinos_diagonales:
                vi, vj = v
                if(not (i,vj) in self.shape_set) and (not (vi,j) in self.shape_set):
                    self.shape_set.add((i,vj))
                    self.boolean_matrix_shape[i,j] = True
                    p_queue.append((i,vj))
                    
    def __obtener_adyacentes_diagonales(self, p: Tuple[int,int]) -> List[Tuple[int,int]]:
        i, j = p
        vecinos_diagonales = [(-1, -1), (1, -1), (-1, 1), (1, 1)]
        return [(i + di, j + dj) for di, dj in vecinos_diagonales if (i + di, j + dj) in self.shape_set]
    
    def __extraer_borde_numpy(self):
        padded = np.pad(self.boolean_matrix_shape, pad_width=1, mode='constant', constant_values=0)
        centro = self.boolean_matrix_shape
        arriba = padded[:-2, 1:-1]
        abajo = padded[2:, 1:-1]
        izquierda = padded[1:-1, :-2]
        derecha = padded[1:-1, 2:]

        erosionada = centro & arriba & abajo & izquierda & derecha
        self.boolean_matrix_shape = centro & ~erosionada
        for i in range(self.height):
            for j in range(self.width):
                if not self.boolean_matrix_shape[i,j]:
                    self.shape_set.discard((i,j))
    
    def __conjuntos_conectados(self) -> List[ConjuntoConectado]:
        toReturn: List[ConjuntoConectado] = []
        p_set_copy = self.shape_set.copy()
        while(p_set_copy):
            p = next(iter(p_set_copy))
            conected_set:ConjuntoConectado = ConjuntoConectado({p},p,p)
            p_set_copy.discard(p)
            conectados = []
            p_queue: deque[Tuple[int, int]] = deque([p])
            while (p_queue):
                p = p_queue.popleft()
                i,j = p
                adyacentes = [(i-1,j), (i,j-1), (i+1,j), (i,j+1)]
                for index in range(len(adyacentes)):
                    if(adyacentes[index] in p_set_copy):
                        p_queue.append(adyacentes[index])
                        conectados.append(adyacentes[index])
                        p_set_copy.discard(adyacentes[index])
                        if index == 1:
                            conected_set.a = conectados[-1]
                        else:
                            conected_set.b = conectados[-1]
            conected_set.set.update(conectados)
            toReturn.append(conected_set)
        toReturn = sorted(toReturn, key=lambda cc: len(cc.set), reverse=True)
        return toReturn
    
    def __tapar_picos_negros(self, factor = 8):
        for i in range(self.height):
            black_row = factor
            for j in range(self.width):
                if(i,j) not in self.shape_set:
                    black_row+=1
                else:
                    if black_row<factor:
                        for n in range(j-1, j-black_row-2, -1):
                            self.shape_set.add((i,n))
                            self.boolean_matrix_shape[i,n] = True
                    black_row = 0

        for j in range(self.width):
            black_row = factor
            for i in range(self.height):
                if(i,j) not in self.shape_set:
                    black_row+=1
                else:
                    if black_row<factor:
                        for n in range(i-1, i-black_row-2, -1):
                            self.shape_set.add((n,j))
                            self.boolean_matrix_shape[n,j] = True
                    black_row = 0
        
    def __eliminar_conjuntos_conectados_pequeños(self, connected_set_list: List[ConjuntoConectado]) -> List[ConjuntoConectado]:
        #delete_factor = (self.height * self.width) / ((self.height + self.width)*22)
        delete_factor = 4
        return [c for c in connected_set_list if len(c.set) >= delete_factor]
    
    def __puntos_en_conjuntos_conectados(self, connected_set_list: List[ConjuntoConectado]) -> set[Tuple[int,int]]:
        toReturn = set()
        for connected_set in connected_set_list:
            toReturn.update(connected_set.set)
        return toReturn
    
    def __conectar_conjuntos(self, connected_set_list: List[ConjuntoConectado]) -> set[Tuple[int,int]]:
        while(len(connected_set_list)>1):
            c1, c2, p1, p2 = self.__conjuntos_y_puntos_mas_cercanos(connected_set_list)
            if(c1.a != p1):
                aux = c1.a
                c1.a = p1
                c1.b = aux
            if(c2.a != p2):
                aux = c2.a
                c2.a = p2
                c2.b = aux
            camino = self.__linea_bresenham(c1.a,c2.a)
            c1.set.update(c2.set)
            c1.set.update(camino)
            c1.a = c2.b
            connected_set_list.remove(c2)
        remaining_set = connected_set_list[0]
        camino = self.__linea_bresenham(remaining_set.a,remaining_set.b)
        remaining_set.set.update(camino)
        return remaining_set.set
    
    #Retorna los dos conjuntos que tengan los puntos a y b mas cercanos. Compara a1 con a2 y b2, y b1 con a2 y b2
    def __conjuntos_y_puntos_mas_cercanos(self, connected_set_list: List[ConjuntoConectado]
    ) -> Tuple[ConjuntoConectado, ConjuntoConectado, Tuple[int, int], Tuple[int, int]]:
        min_dist = float('inf')
        resultado = (None, None, None, None)  # (conj1, conj2, punto1, punto2)
        for i in range(len(connected_set_list)):
            for j in range(i + 1, len(connected_set_list)):
                c1 = connected_set_list[i]
                c2 = connected_set_list[j]
                puntos_c1 = [c1.a, c1.b]
                puntos_c2 = [c2.a, c2.b]
                for p1 in puntos_c1:
                    for p2 in puntos_c2:
                        d = self.__distancia(p1, p2)
                        if d < min_dist:
                            min_dist = d
                            resultado = (c1, c2, p1, p2)
        return resultado

    #Retorna el punto mas cercano a a, tal que sea distinto de a y b.
    def __punto_mas_cercano(self,puntos: set[Tuple[int, int]], a: Tuple[int, int], b: Tuple[int, int]) -> Tuple[int, int]:
        return min((p for p in puntos if p != a and p != b), key=lambda p: self.__distancia(p, a))

    def __distancia(self,p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])
        #return math.hypot(p1[0] - p2[0], p1[1] - p2[1])
    
    #Encuentra el camino de puntos que une dos puntos a y b
    def __linea_bresenham(self,a: Tuple[int, int], b: Tuple[int, int]) -> list[Tuple[int, int]]:
        x0, y0 = a
        x1, y1 = b
        puntos = []

        dx = abs(x1 - x0)
        dy = abs(y1 - y0)

        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1

        err = dx - dy

        while True:
            puntos.append((x0, y0))
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy

        return puntos
            