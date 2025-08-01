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
    matrix_shape: NDArray[np.float64]
    boolean_matrix_shape: NDArray[np.bool_]
    boolean_matrix_border: NDArray[np.bool_]
    background_set: set[Tuple[int,int]]
    shape_set: set[Tuple[int,int]]
    border_set: set[Tuple[int,int]]
    height: int
    width: int

    # Contador para todos los colores
    __color_counter =  Counter()

    def __init__(self, matrix: NDArray[np.float64], delta_threshold = 8):
        self.height, self.width, lab = matrix.shape
        self.matrix = matrix
        self.boolean_matrix_shape = np.zeros((self.height, self.width), dtype=bool)
        self.matrix_shape = np.empty_like(self.matrix)
        self.background_set = set()
        self.shape_set = set()
        self.border_set = set()
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
        self.__calculate_matrix_delta(threshold)
        self.__connect_borders()
        self.__fill_gaps()
        connected_sets = self.__conjuntos_conectados()
        self.__keep_bigger_sets(connected_sets)
        self.__tapar_picos_negros()
        self.__extraer_borde_numpy()
        self.__conectar_diagonales(self.boolean_matrix_shape)
        self.__conectar_diagonales(self.boolean_matrix_border)
        self.shape_set = self.__sync_matrix_set(self.boolean_matrix_shape)
        self.border_set = self.__sync_matrix_set(self.boolean_matrix_border)
        return self.matrix_shape 
    
    def __calculate_matrix_delta(self,threshold):
        for i in range(self.height):
            for j in range(self.width):
                delta = self.__delta_vecinos((i,j))
                if delta>threshold:
                    self.boolean_matrix_shape[i,j] = True
        
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
                left_column_length = 1
            elif left_column_length!=0:
                left_column_length+=1

            if(self.boolean_matrix_shape[i,self.width-1]):
                if(right_column_length != 0):
                    for n in range(i, i - right_column_length, -1):
                        self.boolean_matrix_shape[n,self.width-1] = True
                right_column_length = 1
            elif right_column_length!=0:
                right_column_length+=1

        for j in range(self.width):
            if(self.boolean_matrix_shape[0,j]):
                if(top_row_length != 0):
                    for n in range(j, j - top_row_length, -1):
                        self.boolean_matrix_shape[0,n] = True
                top_row_length = 1
            elif top_row_length!=0:
                top_row_length+=1

            if(self.boolean_matrix_shape[self.height-1,j]):
                if(bottom_row_length != 0):
                    for n in range(j, j - bottom_row_length, -1):
                        self.boolean_matrix_shape[self.height-1,n] = True
                bottom_row_length = 1
            elif bottom_row_length!=0:
                bottom_row_length+=1
    
    def __fill_gaps(self):
        self.shape_set = self.__sync_matrix_set(self.boolean_matrix_shape)
        border_set = self.shape_set.copy()
        while(border_set):
            adyacentes_not_in_set: set[Tuple[int,int]] = self.__obtener_adyacentes_not_in_set(border_set)
            border_set = self.__agregar_cruzados(adyacentes_not_in_set)

    def __obtener_adyacentes_not_in_set(self, border_set: set[Tuple[int,int]]):
        toReturn: set[Tuple[int,int]] = set()
        for i,j in iter(border_set):
            vecinos = [(i-1,j-1), (i,j-1), (i-1,j), (i+1,j), (i,j+1), (i-1,j+1), (i+1,j-1), (i+1,j+1)]
            for vi,vj in vecinos:
                if( 0 <= vi < self.height and 0 <= vj < self.width and not self.boolean_matrix_shape[vi,vj]):
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
                self.boolean_matrix_shape[i,j] = True
                added.add((i, j))

        return added

    #Conecta los puntos que solo estan unidos por diagonales, de forma directa
    def __conectar_diagonales(self, boolean_matrix: NDArray[np.bool_]):
        
        p_queue: deque[Tuple[int, int]] = deque(self.shape_set)
        while(p_queue):
            p = p_queue.popleft()
            i,j = p
            vecinos_diagonales = self.__obtener_adyacentes_diagonales(p, boolean_matrix)
            for v in vecinos_diagonales:
                vi, vj = v
                if(not boolean_matrix[i, vj]) and (not boolean_matrix[vi, j]):
                    boolean_matrix[i,vj] = True
                    p_queue.append((i,vj))
                    
    def __obtener_adyacentes_diagonales(self, p: Tuple[int, int], boolean_matrix: NDArray[np.bool_]) -> List[Tuple[int, int]]:
        i, j = p
        vecinos_diagonales = [(-1, -1), (1, -1), (-1, 1), (1, 1)]
        adyacentes = []

        for di, dj in vecinos_diagonales:
            ni, nj = i + di, j + dj
            if 0 <= ni < self.height and 0 <= nj < self.width:
                if boolean_matrix[ni, nj]:
                    adyacentes.append((ni, nj))

        return adyacentes
    
    def __extraer_borde_numpy(self):
        padded = np.pad(self.boolean_matrix_shape, pad_width=1, mode='constant', constant_values=0)
        centro = self.boolean_matrix_shape
        arriba = padded[:-2, 1:-1]
        abajo = padded[2:, 1:-1]
        izquierda = padded[1:-1, :-2]
        derecha = padded[1:-1, 2:]

        erosionada = centro & arriba & abajo & izquierda & derecha
        self.boolean_matrix_border = centro & ~erosionada
    
    def __conjuntos_conectados(self) -> List[ConjuntoConectado]:
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
                        i,j = p
                        if(i<self.height and j<self.width and matrix_copy[i,j]):
                            matrix_copy[i,j] = False
                            conected_set.set.add(p)
                            p_queue.append((i+1,j))
                            p_queue.append((i,j+1))
                            p_queue.append((i,j-1))
                    toReturn.append(conected_set)
        toReturn = sorted(toReturn, key=lambda cc: len(cc.set), reverse=True)
        return toReturn
    
    def __keep_bigger_sets(self, connected_sets: List[ConjuntoConectado], n=1):
        length = len(connected_sets)
        for n in range(n,length):
            for i,j in connected_sets[n].set:
                self.boolean_matrix_shape[i,j] = False
    
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
                    black_row = 0

    def __sync_matrix_set(self, matrix) -> set[Tuple[int, int]]:
        coords = np.argwhere(matrix)
        return set(map(tuple, coords))
            