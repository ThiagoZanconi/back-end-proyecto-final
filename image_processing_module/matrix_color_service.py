from collections import Counter, deque
from dataclasses import dataclass
import heapq
import time
from typing import List, Tuple
from numpy.typing import NDArray
import numpy as np

from image_processing_module.color_utils import ColorUtils

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
    def __init__(self, matrix: NDArray[np.float64], threshold=8):
        self.height, self.width, lab = matrix.shape
        self.matrix = matrix
        self.boolean_matrix_shape = np.zeros((self.height, self.width), dtype=bool)
        self.boolean_matrix_border = np.zeros((self.height, self.width), dtype=bool)
        self.background_set: set[Tuple[int,int]] = set()
        self.border_set: set[Tuple[int,int]] = set()
        self.shape_set: set[Tuple[int,int]] = set()
        self.background_colors = set()
        self.__color_counter = Counter()
        
        self.__shape_matrix(threshold)
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
    
    def border(self) -> NDArray[np.float64]:
        border_matrix = np.empty_like(self.matrix)
        for i in range(self.height):
            for j in range(self.width):
                if(self.boolean_matrix_border[i,j]):
                    border_matrix[i,j] = [100,0,0]
                else:
                    border_matrix[i,j] = [0,0,0]

        return border_matrix
    
    def border_list(self) -> List[Tuple[int,int]]:
        i, j = -1, -1
        stop = False
        matrix_copy = self.boolean_matrix_border.copy()
        border_list = []
        while(i<self.height-1 and not stop):
            i += 1
            j = -1
            while(j<self.width-1 and not stop):
                j+=1
                if(matrix_copy[i,j]):
                    stop = True
        matrix_copy[i,j] = False
        border_list.append((i,j))
        next: Tuple[int,int]|None = None
        if j + 1 < self.width and matrix_copy[i, j + 1]:
            next = (i, j + 1)
        elif i + 1 < self.height and matrix_copy[i + 1, j]:
            next = (i + 1, j)
        else:
            next = None
        while(next):
            border_list.append(next)
            matrix_copy[next[0],next[1]] = False
            next = self.__get_next(next, matrix_copy)

        return border_list
    
    def matrix_from_list(self, list: List[Tuple[int,int]]) -> NDArray[np.float64]:
        top = bottom = left = right = list[0]
        for i,j in list:
            if(j<left[1]):
                left=(i,j)
            if(j>right[1]):
                right=(i,j)
            if(i<top[0]):
                top=(i,j)
            if(i>bottom[0]):
                bottom=(i,j)
        matriz = np.zeros((bottom[0]+1, right[1]+1, 3), dtype=np.float64)
        for i,j in list:
            matriz[i,j] = [100,0,0]
        return matriz
    
    def find_sub_shape(self, n = 1) -> NDArray[np.float64]:
        delta_pq: List[Tuple[float, Tuple[Tuple[int,int],Tuple[int,int]]]] = self.__delta_pq(self.border_set)
        caminos = []
        for _ in range(n):
            delta, (p,_) = heapq.heappop(delta_pq)
            i, j = p
            first_adyacentes = [(i, j + 1), (i + 1, j), (i, j - 1), (i - 1, j)]
            while(p in self.border_set or p in self.background_set):
                p = first_adyacentes.pop()
            
            camino = [p]
            while(camino[-1] not in self.border_set):
                i, j = camino[-1]
                adyacentes = [(i, j + 1), (i + 1, j + 1), (i + 1, j), (i + 1, j - 1), (i, j - 1), (i - 1, j - 1), (i - 1, j), (i - 1, j + 1)]
                new_delta_pq = self.__delta_pq([(ii,jj) for (ii,jj) in adyacentes if (ii,jj) not in camino and (ii,jj) not in first_adyacentes])
                new_delta, (p,v) = heapq.heappop(new_delta_pq)
                camino.append(v)
                camino.append(p)
            #self.border_set.update(camino)
            caminos.append(camino)
        matrix = self.matrix.copy()
        for i, j in self.shape_set:
            if(tuple(self.matrix[i,j]) == tuple([0,0,0])):
                print("Hay negro")
        for i, j in self.border_set:
            matrix[i,j] = [100,0,0]
        for camino in caminos:
            for i,j in camino:
                matrix[i,j] = [0,128,128]

        return matrix
    
    def get_border_set(self) -> set[Tuple[int,int]]:
        border_set = set()
        for i in range(self.height):
            for j in range(self.width):
                if(self.boolean_matrix_border[i][j]):
                    border_set.add((i,j))
        self.border_set = border_set
        return border_set
    
    def get_main_different_colors(self, n=10, threshold: float = 3.0) -> List[np.uint8]:
        most_common, _ = self.__color_counter.most_common(1)[0]
        main_different_colors = [most_common]
        for color, _ in self.__color_counter.most_common():
            es_diferente = all( ColorUtils.delta_ciede2000(color, c) > threshold for c in main_different_colors )
            if es_diferente:
                main_different_colors.append(color)

            if(len(main_different_colors)==n):
                break
        return main_different_colors
    
    def change_gamma_colors(self, color: List[float], new_color: List[float], threshold: float = 3.0) -> NDArray[np.float64]:
        color_arr = np.array(color, dtype=np.float64)
        new_color_arr = np.array(new_color, dtype=np.float64)
        delta_arr = new_color_arr - color_arr
        toReturn = self.matrix.copy()
        for i in range(self.height):
            for j in range(self.width):
                delta = np.linalg.norm(self.matrix[i, j] - color_arr)
                #delta = ColorUtils.delta_ciede2000(color_arr, self.matrix[i,j])
                if (delta <= threshold):
                    new_val = self.matrix[i, j] + delta_arr

                    new_val[0] = np.clip(new_val[0], 0, 100)      # L
                    new_val[1] = np.clip(new_val[1], -128, 128)   # a
                    new_val[2] = np.clip(new_val[2], -128, 128)   # b

                    toReturn[i, j] = new_val

        return toReturn
    
    def get_point_set_from_color(self, color: List[float], threshold:float) -> set[Tuple[int,int]]:
        color_arr = np.array(color, dtype=np.float64)
        point_set: set[Tuple[int,int]] = set()
        for i in range(self.height):
            for j in range(self.width):
                delta = np.linalg.norm(self.matrix[i, j] - color_arr)
                #delta = ColorUtils.delta_ciede2000(color_arr, self.matrix[i,j])
                if (delta <= threshold):
                    point_set.add((i,j))
        return point_set
    
    def __delta_pq(self, points: set[Tuple[int,int]]) -> List[Tuple[float, Tuple[Tuple[int,int],Tuple[int,int]]]]:
        delta_pq: List[Tuple[float, Tuple[Tuple[int,int],Tuple[int,int]]]] = []
        for i, j in points:
            delta_acum, max_vecino = self.__delta_neighbors((i,j))
            heapq.heappush(delta_pq, (-delta_acum, ((i,j),max_vecino)))
        return delta_pq
    
    def __delta_acum(self, p:Tuple[int,int]) -> float:
        i, j = p
        adyacentes = [(i, j + 1), (i + 1, j + 1), (i + 1, j), (i + 1, j - 1), (i, j - 1), (i - 1, j - 1), (i - 1, j), (i - 1, j + 1)]
        adyacentes = [(ii,jj) for (ii,jj) in adyacentes if (ii,jj) not in self.background_set and (ii,jj) not in self.border_set]
        delta_acum = 0.0
        for ii, jj in adyacentes:
            delta_acum += ColorUtils.delta_ciede2000(self.matrix[i,j], self.matrix[ii,jj])
        return delta_acum

    def __get_next(self, p, boolean_matrix) -> Tuple[int, int] | None:
        i, j = p
        adyacentes = [(i, j + 1), (i + 1, j + 1), (i + 1, j), (i + 1, j - 1), (i, j - 1), (i - 1, j - 1), (i - 1, j), (i - 1, j + 1)]
        for ai, aj in adyacentes:
            if 0 <= ai < self.height and 0 <= aj < self.width:
                if boolean_matrix[ai, aj]:
                    return (ai, aj)
        return None

    def __shape_matrix(self, threshold) -> NDArray[np.float64]:
        start = time.perf_counter()
        self.__calculate_matrix_delta(threshold)
        end_delta = time.perf_counter()
        print(f"Tiempo de ejecución: {end_delta - start:.6f} segundos")
        self.__connect_borders()
        self.__fill_gaps()
        connected_sets = self.__connected_sets()
        self.__keep_bigger_sets(connected_sets)
        self.__extract_border()
        end = time.perf_counter()
        print(f"Tiempo de ejecución: {end - start:.6f} segundos")
    
    def __calculate_matrix_delta(self,threshold):
        for i in range(self.height):
            for j in range(self.width):
                delta, _ = self.__delta_neighbors((i,j))
                if delta>threshold:
                    self.boolean_matrix_shape[i,j] = True
                    self.shape_set.add((i,j))
        
    def __delta_neighbors(self, p: Tuple[int, int]) -> Tuple[np.float64, Tuple[int,int]]:
        i, j = p
        max_vecino = (0,0)
        max_delta = 0.0
        vecinos = [ (i,j-1), (i-1,j), (i+1,j), (i,j+1)]
        for ii, jj in vecinos:
            if 0 <= ii < self.height and 0 <= jj < self.width:
                delta = np.linalg.norm(self.matrix[i, j] - self.matrix[ii, jj])
                #delta = ColorUtils.delta_ciede2000(self.matrix[i, j], self.matrix[ni, nj])
                if(delta>max_delta):
                    max_delta = delta
                    max_vecino = ii, jj

        return max_delta, max_vecino
    
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

    def __fill_gaps(self):
        visited = np.zeros((self.height, self.width), dtype=bool)
        shape_group = set()
        background_group = set()

        for i in range(self.height):
            for j in range(self.width):
                if not self.boolean_matrix_shape[i, j] and not visited[i, j]:
                    # Nuevo grupo de fondo potencial
                    group = set()
                    queue = deque([(i, j)])
                    visited[i, j] = True
                    closed = True  # asumimos cerrado, si tocamos borde => no cerrado

                    while queue:
                        x, y = queue.popleft()
                        group.add((x, y))

                        for nx, ny in [(x-1,y), (x+1,y), (x,y-1), (x,y+1)]:
                            if 0 <= nx < self.height and 0 <= ny < self.width:
                                if not visited[nx, ny] and not self.boolean_matrix_shape[nx, ny]:
                                    visited[nx, ny] = True
                                    queue.append((nx, ny))
                            else:
                                closed = False

                    # Clasificación final del grupo
                    if closed:
                        shape_group.update(group)
                    else:
                        background_group.update(group)

        # Actualizar matriz
        for i, j in shape_group:
            self.boolean_matrix_shape[i, j] = True
            self.shape_set.add((i, j))
        self.background_set.update(background_group)

    def __extract_border(self):
        for i in range(self.height):
            for j in range(self.width):
                adyacentes = [(i, j + 1), (i + 1, j + 1), (i + 1, j), (i + 1, j - 1), (i, j - 1), (i - 1, j - 1), (i - 1, j), (i - 1, j + 1)]
                if(not self.boolean_matrix_shape[i,j]):
                    if(self.shape_set.intersection(adyacentes)):
                        self.border_set.add((i,j))
                        self.boolean_matrix_border[i,j] = True
                    else:
                        self.background_set.add((i,j))
    
    def __connected_sets(self) -> List[ConjuntoConectado]:
        toReturn: List[ConjuntoConectado] = []
        matrix_copy = self.boolean_matrix_shape.copy()
        for i in range(self.height):
            for j in range(self.width):
                if matrix_copy[i,j]:
                    p = (i,j)
                    p_queue: deque[Tuple[int, int]] = deque([p])
                    conected_set:ConjuntoConectado = ConjuntoConectado({p},p,p)
                    while p_queue:
                        p = p_queue.popleft()
                        ii,jj = p
                        if(ii<self.height and jj<self.width and matrix_copy[ii,jj]):
                            matrix_copy[ii,jj] = False
                            conected_set.set.add(p)
                            p_queue.append((ii+1,jj))
                            p_queue.append((ii,jj+1))
                            p_queue.append((ii,jj-1))
                            p_queue.append((ii-1,jj))
                    toReturn.append(conected_set)
        toReturn = sorted(toReturn, key=lambda cc: len(cc.set), reverse=True)
        return toReturn
    
    def __keep_bigger_sets(self, connected_sets: List[ConjuntoConectado], n=1):
        length = len(connected_sets)
        for n in range(n,length):
            for i,j in connected_sets[n].set:
                self.boolean_matrix_shape[i,j] = False
                self.shape_set.discard((i,j))

    def __pulir_shape(self):
        for i in range(self.height):
            for j in range(self.width):
                if (i,j) not in self.shape_set:
                    self.background_colors.add(tuple(self.matrix[i,j]))
        copy = self.shape_set.copy()
        for i, j in copy:
            if (tuple(self.matrix[i,j]) in self.background_colors):
                self.boolean_matrix_shape[i,j] = False
                self.shape_set.discard((i,j))
    
    def __tapar_picos_negros(self, factor = 8):
        for i in range(self.height):
            black_row = factor
            for j in range(self.width):
                if not self.boolean_matrix_shape[i,j]:
                    black_row+=1
                else:
                    if black_row<factor:
                        for n in range(j-1, j-black_row-2, -1):
                            self.boolean_matrix_shape[i,n] = True
                            self.shape_set.add((i,n))
                    black_row = 0

        for j in range(self.width):
            black_row = factor
            for i in range(self.height):
                if not self.boolean_matrix_shape[i,j]:
                    black_row+=1
                else:
                    if black_row<factor:
                        for n in range(i-1, i-black_row-2, -1):
                            self.boolean_matrix_shape[n,j] = True
                            self.shape_set.add((n,j))
                    black_row = 0

    def __sync_matrix_set(self, matrix) -> set[Tuple[int, int]]:
        coords = np.argwhere(matrix)
        return set(map(tuple, coords))
            