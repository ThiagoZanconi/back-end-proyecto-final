import heapq
import math
from typing import Dict, List, Tuple

class ShapeAnalyzerService:
    shape_list: List[List[Tuple[int,int]]]
    shape_segment_list: List[List[Tuple[Tuple[int,int],Tuple[int,int]]]]
    segment_point_map_list: List[ Dict[Tuple[int,int], List[Tuple[int,int]]] ]

    def __init__(self, shapes: List[List[Tuple[int,int]]], n = 20):
        self.shape_list = shapes
        self.shape_segment_list = []
        self.segment_point_map_list = []
        self.__descomponer_rectas(n)
        #self.__analyze()

    def __analyze(self):
        for shape in self.shape_list:
            directions = []
            chain_code = []
            for (x1, y1), (x2, y2) in zip(shape, shape[1:] + [shape[0]]):  # circular
                dx = x2 - x1
                dy = y2 - y1
                directions.append((dx, dy))
                if (dx, dy) == (0, 1): chain_code.append(0)  # derecha
                elif (dx, dy) == (1, 1): chain_code.append(1)  # abajo, derecha
                elif (dx, dy) == (1, 0): chain_code.append(2)  # abajo
                elif (dx, dy) == (1, -1): chain_code.append(3)  # abajo, izquierda
                elif (dx, dy) == (0, -1): chain_code.append(4)  # izquierda
                elif (dx, dy) == (-1, -1): chain_code.append(5)  # arriba, izquierda
                elif (dx, dy) == (-1, 0): chain_code.append(6)  # arriba
                elif (dx, dy) == (-1, 1): chain_code.append(7)  # arriba, derecha

    def  __descomponer_rectas(self, n):
        for shape in self.shape_list:
            rectas_pq: List[Tuple[int, Tuple[int, int]]] = []
            segmentos_agregados: set[Tuple[int,int]] = set()
            segment_point_map: Dict[ Tuple[int,int], List[Tuple[int,int]]] = {}
            restantes: List[int] = []
            to_be_connected: List[int] = []
            length = len(shape)
            for i in range(length):
                restantes.append(i)

            while(restantes):
                candidatos: List[Tuple[int,int,float]] = []
                for i in restantes:
                    last_not_used = n + i
                    while (last_not_used % length not in restantes):
                        last_not_used -= 1
                        if last_not_used == -1:
                            last_not_used = length - 1

                    segment_length = (last_not_used - i) % length
                    delta = self._comparar_angulo_rectas(shape[i], shape[(i + (segment_length // 2)) % length], shape[last_not_used%length])
                    extremos = ((i, (i + segment_length) % length))
                    if(segment_length!=0 and extremos not in segmentos_agregados):
                        candidatos.append((i,segment_length,delta))

                candidatos.sort(key=lambda x: x[2])
                i,segment_length,delta = candidatos.pop(0)
                heapq.heappush(rectas_pq, (i, (shape[i], shape[(i + segment_length) % length])))
                extremos = ((i, (i + segment_length) % length))
                segmentos_agregados.add(extremos)
                indices = [(i + j + 1) % length for j in range(segment_length-1)]
                point_list: List[Tuple[int,int]] = [i]
                for ii in indices:
                    restantes.remove(ii)
                    point_list.append(shape[ii])
                point_list.append(shape[(i + segment_length) % length])
                segment_point_map[(i, (i + segment_length) % length)] = point_list
                for extremo in [i, (i + segment_length) % length]:
                    if extremo in to_be_connected:
                        to_be_connected.remove(extremo)
                        restantes.remove(extremo)
                    else:
                        to_be_connected.append(extremo)
            rectas = [heapq.heappop(rectas_pq)[1] for _ in range(len(rectas_pq))]
            self.segment_point_map_list.append(segment_point_map)
            self.shape_segment_list.append(rectas)
            
    def _comparar_angulo_rectas(self, p1:Tuple[int,int], p2: Tuple[int,int], p3:Tuple[int,int]) -> float:
        x1,y1 = p1
        x2,y2 = p2
        x3,y3 = p3
        a12 = math.atan2(y2-y1, x2-x1) 
        a13 = math.atan2(y3-y1, x3-x1)
        return self.__comparar_angulos(a12,a13)
    
    #Delta entre [0, π/2]
    def __comparar_angulos(self, angle1, angle2) -> float:
        delta = abs(angle1 - angle2)
        delta = min(delta, 2*math.pi - delta)

        # Si querés ignorar el sentido
        if delta > math.pi:
            delta = 2*math.pi - delta
        return delta
    
    def corrimiento_circular_inplace(lista: List, i:int):
        lista[:] = lista[i:] + lista[:i]
