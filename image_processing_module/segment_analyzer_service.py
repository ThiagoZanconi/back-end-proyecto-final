from math import sqrt
from typing import List, Tuple

class SegmentAnalyzerService:
    shape_segment_list: List[List[Tuple[Tuple[int,int],Tuple[int,int]]]]
    grouped_shapes_segments: List[List[List[Tuple[ Tuple[int,int], Tuple[int,int] ]]]] = []

    def __init__(self, shape_segment_list: List[List[Tuple[Tuple[int,int],Tuple[int,int]]]], n: int = 15):
        self.shape_segment_list = shape_segment_list
        self.__group_segments_by_length(n)

    def __group_segments_by_length(self, n):
        for shape_segment in self.shape_segment_list:
            shape_length = self.__shape_length(shape_segment)
            target = shape_length / n

            grupos: List[List[Tuple[ Tuple[int,int], Tuple[int,int] ]]] = []
            grupo_actual: List[Tuple[ Tuple[int,int], Tuple[int,int] ]] = []
            acum = 0.0

            for seg in shape_segment:
                l = self.__segment_length(seg)
                if acum + l > target and len(grupos) < n - 1:
                    grupos.append(grupo_actual)
                    grupo_actual = []
                    acum = 0.0
                grupo_actual.append(seg)
                acum += l

            grupos.append(grupo_actual)  # último grupo
            self.grouped_shapes_segments.append(grupos)

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