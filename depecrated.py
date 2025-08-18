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


        p_queue = deque(point for par in last_sets for point in par.set)
        self.__join_neighbors(p_queue)
     
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

'''