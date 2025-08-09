from dataclasses import dataclass
from math import sqrt
import random
from typing import List, Tuple
import numpy as np
from shape_analyzer_service import Segment

@dataclass
class Delta:
    dx: float
    dy: float

class SegmentAnalyzerService:
    #Primera lista (shapes). Segunda lista (segmentos)
    shape_segment_list: List[List[Segment]]
    #Primera lista (shapes). Segunda lista (n grupos por distancia). Tercera lista (segmentos)
    grouped_shapes_segments: List[List[List[Segment]]] = []
    #Primer lista de n grupos. Segunda lista de grupos de segmentos que tienen extremos parecidos (en terminos relativos). Tercera lista (segmentos).
    similar_segments:list[list[List[Segment]]] = []
    shapes_grouped_segments_delta_matrix: List[np.ndarray] = []
    n:int

    def __init__(self, shape_segment_list: List[List[Segment]], n: int = 10):
        self.shape_segment_list = shape_segment_list
        self.n = n
        self.__group_segments_by_length()
        self.__compare_segmet_groups_extremes_relative_distance()
        self.__fill_delta_matrix()

    def new_shape(self) -> List[Segment]:
        new_shape: List[Segment] = []
        for i in range(self.n):
            print(self.similar_segments[i])
            #random_segment_group = random.choice(self.similar_segments[i])
            random_segment_group = self.similar_segments[i][0]
            new_shape.extend(random_segment_group)
            print("-----")
        return new_shape
    
    def new_shape_v2(self) -> List[List[Segment]]:
        new_shape: List[List[Segment]] = []
        shapes_size = len(self.grouped_shapes_segments)
        index = random.randint(0, shapes_size)
        #Starting from random shape
        current_shape_groups = self.grouped_shapes_segments[index]
        shape_groups_size = len(current_shape_groups)
        while(idx<shape_groups_size):
            shape_group = current_shape_groups[idx]
            found = False
            for n in range(shapes_size):
                new_shape_groups = self.grouped_shapes_segments[n]
                if(new_shape_groups != current_shape_groups):
                    p1 = shape_group[0].first
                    p2 = shape_group[-1].last
                    p3 = new_shape_groups[idx][0].first
                    p4 = new_shape_groups[idx][-1].last
                    if(self.__similar_segments(p1, p2, p3, p4)):
                        found = True
                        current_shape_groups = new_shape_groups
                        break
            if(not found):
                new_shape.append(shape_group)
            idx+=1

        return new_shape
    
    def similar_shape_groups(self, treshold = 50) -> List[Tuple[List[Segment], List[Segment]]]:
        height, width = self.shapes_grouped_segments_delta_matrix[0].shape[:2]
        toReturn = []
 
        for n in range(height):
            total_delta = 0.0
            for i in range(height):
                for j in range(width):
                    total_delta += abs(self.shapes_grouped_segments_delta_matrix[0][i][j].dx - self.shapes_grouped_segments_delta_matrix[1][i][j].dx)
            if (total_delta < treshold):
                toReturn.append((self.grouped_shapes_segments[0][n], self.grouped_shapes_segments[1][n]))
        return toReturn

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

    def __fill_delta_matrix(self):
        for shape_groups in self.grouped_shapes_segments:
            n = len(shape_groups)
            delta_matrix: np.ndarray = np.empty((n, n), dtype=object)
            for i in range(n):
                for j in range(i, n):
                    s1 = Segment([], shape_groups[i][0].first, shape_groups[i][-1].last)
                    s2 = Segment([], shape_groups[j][0].first, shape_groups[j][-1].last)
                    d: Delta = self.__segments_delta(s1, s2)
                    delta_matrix[i, j] = d
                    delta_matrix[j, i] = d 
            self.shapes_grouped_segments_delta_matrix.append(delta_matrix)

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

    def __segments_delta(self, s1: Segment, s2: Segment) -> Delta:
        m1x = (s1.first[0] + s1.last[0]) / 2
        m1y = (s1.first[1] + s1.last[1]) / 2
        m2x = (s2.first[0] + s2.last[0]) / 2
        m2y = (s2.first[1] + s2.last[1]) / 2
        dx = m2x - m1x
        dy = m2y - m1y
        return Delta(dx,dy)
                 