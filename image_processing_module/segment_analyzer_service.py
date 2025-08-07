from math import sqrt
import math
import random
from typing import List, Tuple

class SegmentAnalyzerService:
    shape_segment_list: List[List[Tuple[Tuple[int,int],Tuple[int,int]]]]
    grouped_shapes_segments: List[List[List[Tuple[ Tuple[int,int], Tuple[int,int] ]]]] = []
    segment_group_shape_anlge_length_list: List[List[Tuple[float,float]]] = []
    DELTA_THRESHOLD = 0.2
    n:int = 15

    def __init__(self, shape_segment_list: List[List[Tuple[Tuple[int,int],Tuple[int,int]]]], n: int = 15):
        self.shape_segment_list = shape_segment_list
        self.n = n
        self.__group_segments_by_length()

    def __group_segments_by_length(self):
        for shape_segment in self.shape_segment_list:
            shape_length = self.__shape_length(shape_segment)
            target = shape_length / self.n

            grupos: List[List[Tuple[ Tuple[int,int], Tuple[int,int] ]]] = []
            grupo_actual: List[Tuple[ Tuple[int,int], Tuple[int,int] ]] = []
            acum = 0.0

            for seg in shape_segment:
                l = self.__segment_length(seg)
                if acum + l > target and len(grupos) < self.n - 1:
                    grupos.append(grupo_actual)
                    grupo_actual = []
                    acum = 0.0
                grupo_actual.append(seg)
                acum += l

            grupos.append(grupo_actual)  # último grupo
            self.grouped_shapes_segments.append(grupos)

        #Calcula para grupo de segmentos el largo y el angulo del segmento resultante entre el primer y ultimo punto del grupo de segmentos.
    def __calculate_segment_group_angle_and_length(self):
        for grouped_shape_segments in self.grouped_shapes_segments:
            segment_group_anlge_length_list: List[Tuple[float,float]] = []
            for grouped_segments in grouped_shape_segments:
                p1 = grouped_segments[0][0]
                p2 = grouped_segments[-1][1]
                angle = math.atan2(p2[1]-p1[1], p2[0]-p1[0])
                length = self.__segment_length(p1,p2)
                segment_group_anlge_length_list.append((angle,length))
            self.segment_group_shape_anlge_length_list.append(segment_group_anlge_length_list)

    def __compare_angle_segment_length(self):
        similar_segments = []
        for i in range(self.n):
            segment_groups_i = [[self.segment_group_shape_anlge_length_list[0]]]
            for segment_group_shape_anlge_length in self.segment_group_shape_anlge_length_list[1:]:
                segment = segment_group_shape_anlge_length[i]
                inserted = False
                for segment_group in segment_groups_i:
                    r = random.choice(segment_group)
                    if (self.__similar_segments(segment, r)):
                        segment_group.append(segment)
                        inserted = True
                        break
                if not inserted:
                    segment_groups_i.append([segment])
            similar_segments.append(max(segment_groups_i, key=len))
                        

    def __shape_length(self, shape_segments: List[Tuple[ Tuple[int,int], Tuple[int,int]]]) -> float:
        length = 0.0
        for a,b in shape_segments:
            length+= self.__segment_length(a,b)
        return length

    def __segment_length(self, a: Tuple[int,int], b: Tuple[int,int]) -> float:
        ax, ay = a
        bx, by = b
        dx = bx-ax
        dy = by-ay
        return sqrt(dx**2 + dy**2)
    
    #Tuple[angle,length]
    #Retorna true si son similares los segmentos
    def __similar_segments(self, s1: Tuple[float,float], s2: Tuple[float,float]):
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
                
                     
                 