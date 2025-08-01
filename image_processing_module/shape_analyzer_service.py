import math
from typing import List, Tuple

class ShapeAnalyzerService:
    shapes: List[List[Tuple[int,int]]]
    shape_lines: 

    def __init__(self, shapes: List[List[Tuple[int,int]]]):
        self.shapes = shapes
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

    def  __descomponer_rectas(self, n = 20):
        for shape in self.shapes:
            i = 0
            rectas = []
            length = len(shape)
            factor = length//20
            while(len(rectas)<20):
                for j in range(factor//2):
                    x1,y1 = shape[(i+j)%length]
                    x2,y2 = shape[((i+j)+factor//4) % length]
                    x3,y3 = shape[((i+j)+factor) % length]
                    a12 = math.atan2(y2-y1, x2-x1) 
                    a13 = math.atan2(y3-y1, x3-x1)
                    delta = self.__comparar_angulos(a12,a13)
                i+=1

    def __comparar_angulos(self, angle1, angle2):
        delta = abs(angle1 - angle2)
        delta = min(delta, 2*math.pi - delta)

        # Si querés ignorar el sentido
        if delta > math.pi:
            delta = 2*math.pi - delta
        return delta