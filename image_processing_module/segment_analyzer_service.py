from math import sqrt
import random
from typing import List, Tuple

from shape_analyzer_service import Segment

class SegmentAnalyzerService:
    #Primera lista (shapes). Segunda lista (segmentos)
    shape_segment_list: List[List[Segment]]
    #Primera lista (shapes). Segunda lista (n grupos por distancia). Tercera lista (segmentos)
    grouped_shapes_segments: List[List[List[Segment]]] = []
    #Primer lista de n grupos. Segunda lista de grupos de segmentos que tienen extremos parecidos (en terminos relativos). Tercera lista (segmentos).
    similar_segments:list[list[List[Segment]]] = []
    n:int

    def __init__(self, shape_segment_list: List[List[Segment]], n: int = 15):
        self.shape_segment_list = shape_segment_list
        self.n = n
        self.__group_segments_by_length()
        self.__compare_segmet_groups_extremes_relative_distance()

    def new_shape(self) -> List[Segment]:
        new_shape: List[Segment] = []
        for i in range(self.n):
            print(self.similar_segments[i])
            #random_segment_group = random.choice(self.similar_segments[i])
            random_segment_group = self.similar_segments[i][0]
            new_shape.extend(random_segment_group)
            print("-----")
        return new_shape

    def __group_segments_by_length(self):
        for segment_list in self.shape_segment_list:
            shape_length = self.__shape_length(segment_list)
            target = shape_length / self.n

            grupos: List[List[Segment]] = []
            grupo_actual: List[Segment] = []
            acum = 0.0

            for seg in segment_list:
                l = self.__segment_length(seg.first, seg.last)
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
                random.shuffle(segment_groups_i)
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
        tolerancia = 0.20  # 10%

        if (self.__diferencia_relativa(dx1, dx2) <= tolerancia and self.__diferencia_relativa(dy1, dy2) <= tolerancia):
            return True
        else:
            return False
    
    def __diferencia_relativa(self, a, b):
        if a == 0:
            return abs(b) < 1e-9  # o considerar esto como 100% si a es cero
        return abs(a - b) / abs(a)            
                 