from dataclasses import dataclass
from math import sqrt
import random
from typing import List, Tuple
from shape_analyzer_service import Segment

@dataclass
class Delta:
    dx: float
    dy: float

class SegmentAnalyzerService:
    #Primera lista (shapes). Segunda lista (segmentos)
    shape_segment_list: List[List[Segment]]
    shape_size: int

    def __init__(self, shape_segment_list: List[List[Segment]]):
        self.shape_segment_list = shape_segment_list
        self.__match_shape_segment_size()

    def __match_shape_segment_size(self):
        min_shape_size = min(len(sublista) for sublista in self.shape_segment_list)
        for shape in self.shape_segment_list:
            shape_size = len(shape)
            if(shape_size > min_shape_size):
                self.__join_smallest_segments(shape,shape_size-min_shape_size)
        self.shape_size = min_shape_size
    
    def new_shape(self) -> List[Segment]:
        new_segment_list: List[Segment] = []
        shapes_size = len(self.shape_segment_list)
        index = random.randint(0, shapes_size - 1)
        #Starting from random shape
        current_shape = self.shape_segment_list[index]
        idx = 0
        while(idx<self.shape_size):
            current_segment = current_shape[idx]
            found = False
            for n in range(shapes_size):
                new_shape = self.shape_segment_list[n]
                if(new_shape != current_shape):
                    p1 = current_segment.first
                    p2 = current_segment.last
                    p3 = new_shape[idx].first
                    p4 = new_shape[idx].last
                    if(self.__similar_segments(p1, p2, p3, p4)):
                        found = True
                        new_segment_list.append(new_shape[idx])
                        current_shape = new_shape
                        break
            if(not found):
                new_segment_list.append(current_segment)
            idx+=1
        return new_segment_list
    
    def __join_smallest_segments(self, shape: List[Segment], n:int):
        while(n):
            length = self.__segment_length(shape[0].first,shape[0].last) + self.__segment_length(shape[1].first,shape[1].last)
            shortest_tuple = 0
            for i in range(1,len(shape)-1):
                new_length = self.__segment_length(shape[i].first,shape[i].last) + self.__segment_length(shape[i+1].first,shape[i+1].last) 
                if new_length < length:
                    shortest_tuple = i
                    length = new_length
            self.__concatenate_segments(shape, shortest_tuple)
            n -= 1

    def __segment_length(self, a: Tuple[int,int], b: Tuple[int,int]) -> float:
        ax, ay = a
        bx, by = b
        dx = bx-ax
        dy = by-ay
        return sqrt(dx**2 + dy**2)
    
    def __concatenate_segments(self, shape: List[Segment], i:int):
        first = shape[i].first
        last = shape[i+1].last
        point_list = shape[i].points + shape[i+1].points
        new_segment = Segment(point_list, first, last)
        shape.pop(i)
        shape.pop(i)
        shape.insert(i,new_segment)
    
    def __similar_segments(self, p1:Tuple[int,int], p2:Tuple[int,int] ,p3:Tuple[int,int] ,p4:Tuple[int,int] ) -> bool:
        x1,y1 = p1
        x2,y2 = p2
        x3,y3 = p3
        x4,y4 = p4
        dx1, dy1 = x2 - x1, y2 - y1
        dx2, dy2 = x4 - x3, y4 - y3

        return dx1 == dx2 and dy1 == dy2
                 