import math
from typing import List, Tuple

class ShapeAnalyzerService:
    shapes: List[List[Tuple[int,int]]]
    ANGLE_THRESHOLD = 0.175
    shape_lines: List[List[Tuple[Tuple[int,int],Tuple[int,int]]]]

    def __init__(self, shapes: List[List[Tuple[int,int]]]):
        self.shapes = shapes
        self.shape_lines = []
        self.__descomponer_rectas()
        #self.__analyze()

    def __analyze(self):
        for shape in self.shapes:
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

    def  __descomponer_rectas(self, n = 30):
        for shape in self.shapes:
            rectas: List[Tuple[int,int]] = []
            restantes: List[int] = []
            length = len(shape)
            
            for i in range(length):
                restantes.append(i)
            while(restantes):
                candidatos: List[Tuple[int,int,float]] = []
                print("Restantes: ",restantes)
                for index in range(len(restantes)):
                    i = restantes[index]
                    last_not_used = n + i
                    while(last_not_used%length not in restantes):
                        last_not_used-=1
                        if(last_not_used == -1):
                            last_not_used = length -1
                    x1,y1 = shape[i]
                    x2,y2 = shape[((i+last_not_used)//2)%length]
                    x3,y3 = shape[(last_not_used)%length]
                    a12 = math.atan2(y2-y1, x2-x1) 
                    a13 = math.atan2(y3-y1, x3-x1)
                    delta = self.__comparar_angulos(a12,a13)
                    segment_length = last_not_used - i
                    if(last_not_used<i):
                        segment_length += length
                    candidatos.append((i,segment_length,delta))
                candidatos.sort(key=lambda x: x[2])
                delta:float = 0.0
                while(delta < self.ANGLE_THRESHOLD and candidatos):
                    i,segment_length,delta = candidatos.pop(0)
                    if(delta < self.ANGLE_THRESHOLD):
                        rectas.append((shape[i], shape[(i+n)%length]))
                        indices = [(i + j + 1) % length for j in range(segment_length)]
                        # Para evitar errores al modificar mientras iterás, creás nueva lista
                        candidatos = [(i,segment_length,delta) for i,segment_length, delta in candidatos if i not in indices]
                        restantes = [i for i in restantes if i not in indices]
                n = max(1, math.floor(n * 0.9))
                if n <= 2:
                    break
            self.shape_lines.append(rectas)
            
    #Delta entre [0, π/2]
    def __comparar_angulos(self, angle1, angle2) -> float:
        delta = abs(angle1 - angle2)
        delta = min(delta, 2*math.pi - delta)

        # Si querés ignorar el sentido
        if delta > math.pi:
            delta = 2*math.pi - delta
        return delta