from math import sqrt
import math
import random
from typing import List, Tuple

from image_processing_module.shape_analyzer_service import Segment

class SegmentAnalyzerService:
    #Primera lista (shapes). Segunda lista (segmentos)
    shape_segment_list: List[List[Segment]]
    #Primera lista (shapes). Segunda lista (n grupos por distancia). Tercera lista (segmentos)
    grouped_shapes_segments: List[List[List[Segment]]] = []
    #Primer lista de n grupos. Segunda lista de grupos de segmentos que tienen extremos parecidos (en terminos relativos). Tercera lista (segmentos).
    similar_segments:list[list[List[Tuple[int,int]]]] = []
    n:int

    def __init__(self, shape_segment_list: List[List[Segment]], n: int = 15):
        self.shape_segment_list = shape_segment_list
        self.n = n
        self.__group_segments_by_length()
        self.__compare_segmet_groups_extremes_relative_distance()

    def __group_segments_by_length(self):
        for segment_list in self.shape_segment_list:
            shape_length = self.__shape_length(segment_list)
            target = shape_length / self.n

            grupos: List[List[Segment]] = []
            grupo_actual: List[Segment] = []
            acum = 0.0

            for seg in segment_list:
                l = self.__segment_length(seg)
                if acum + l > target and len(grupos) < self.n - 1:
                    grupos.append(grupo_actual)
                    grupo_actual = []
                    acum = 0.0
                grupo_actual.append(seg)
                acum += l

            grupos.append(grupo_actual)  # último grupo
            self.grouped_shapes_segments.append(grupos)  

    def __compare_segmet_groups_extremes_relative_distance(self):
        for i in range(self.n):
            segment_groups_i = [[self.grouped_shapes_segments[0][i]]]
            for segment_shape_group in self.grouped_shapes_segments[1:]:
                current_segment_group = segment_shape_group[i]
                p1 = current_segment_group[0].first
                p2 = current_segment_group[-1].last
                inserted = False
                for segment_group in segment_groups_i:
                    r = random.choice(segment_group)
                    p3 = r[0].first
                    p4 = r[-1].last
                    if (self.__similar_segments(p1, p2, p3, p4)):
                        segment_group.append(current_segment_group)
                        inserted = True
                        break
                if not inserted:
                    segment_groups_i.append([current_segment_group])
            self.similar_segments.append(max(segment_groups_i, key=len))                     

    def __shape_length(self, segment_list: List[Segment]) -> float:
        length = 0.0
        for segment in segment_list:
            length+= self.__segment_length(segment.first,segment.last)
        return length

    def __segment_length(self, a: Tuple[int,int], b: Tuple[int,int]) -> float:
        ax, ay = a
        bx, by = b
        dx = bx-ax
        dy = by-ay
        return sqrt(dx**2 + dy**2)
    
    def __similar_segments(self, p1:Tuple[int,int], p2:Tuple[int,int] ,p3:Tuple[int,int] ,p4:Tuple[int,int] ) -> bool:
        x1,y1 = p1
        x2,y2 = p2
        x3,y3 = p3
        x4,y4 = p4
        dx1, dy1 = x2 - x1, y2 - y1
        dx2, dy2 = x4 - x3, y4 - y3
        tolerancia = 0.10  # 10%

        if (self.__diferencia_relativa(dx1, dx2) <= tolerancia and self.__diferencia_relativa(dy1, dy2) <= tolerancia):
            return True
        else:
            return False
    
    def __diferencia_relativa(self, a, b):
        if a == 0:
            return abs(b) < 1e-9  # o considerar esto como 100% si a es cero
        return abs(a - b) / abs(a)

    #Tuple[angle,length]
    #Retorna true si son similares los segmentos
    def __similar_segments_v2(self, s1: Tuple[float,float], s2: Tuple[float,float]):
        a1, l1 = s1
        a2, l2 = s2
        base = max(l1, l2, 1e-6)  # Evita división por cero
        return self.__comparar_angulos(a1,a2) < self.DELTA_THRESHOLD and abs(l1 - l2) / base < 0.1

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
                
                     
                 