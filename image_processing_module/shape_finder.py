from typing import List, Tuple
from numpy.typing import NDArray
import numpy as np
from color_utils import ColorUtils

class ShapeFinder:
    @staticmethod
    def find_shape(original_matrix: NDArray[np.float64], n: int = 3) -> NDArray[np.float64]:
        height, width, _ = original_matrix.shape
        si, sj = ColorUtils.get_starting_point(original_matrix)
        si += 1
        current_path: List[Tuple[int, int]] = [(si, sj)]
        finish = False

        while not finish:
            current_path, delta = ShapeFinder.__find_best_path(original_matrix, current_path, n - 1)
            finish = ShapeFinder.__adjacent_points_in_path(current_path,height,width)

        # Pintar la forma en rojo
        for point in current_path:
            print(f"{point[0]},{point[1]}")
            original_matrix[point[0], point[1]] = [100, 0, 0]

        return original_matrix

    @staticmethod
    def __find_best_path(original_matrix: NDArray[np.float64], current_path: List[Tuple[int, int]], n: int) -> Tuple[List[Tuple[int, int]], float]:
        
        si = current_path[-1][0]
        sj = current_path[-1][1]

        if n == 0 or tuple(original_matrix[si,sj]) == (0,0,0):
            return current_path, 0.0

        height, width, _ = original_matrix.shape
        paths = []

        for i in range(si - 1, si + 2):
            for j in range(sj - 1, sj + 2):
                if (0 <= i < height) and (0 <= j < width) and ((i, j) not in current_path):
                    new_path = current_path + [(i, j)]
                    print(i)
                    print(j)
                    sub_path, sub_delta = ShapeFinder.__find_best_path(
                        original_matrix, new_path, n - 1
                    )

                    delta = ColorUtils.delta_ciede2000(original_matrix[si, sj], original_matrix[i, j])
                    total_delta = sub_delta + delta
                    paths.append((sub_path, total_delta))

        if paths:
            best_path, best_delta = min(paths, key=lambda x: x[1])
            return best_path, best_delta
        else:
            return current_path, 0.0
        
    def __adjacent_points_in_path(current_path,height,width) -> bool:
        toReturn = False
        si = current_path[-1][0]
        sj = current_path[-1][1]
        adjacent_points_in_path = 0
        for i in range(si - 1, si + 2):
            for j in range(sj - 1, sj + 2):
                if (0 <= i < height) and (0 <= j < width) and ((i, j) != (si, sj)):
                    if (i, j) in current_path:
                        adjacent_points_in_path += 1

        if adjacent_points_in_path >= 2:
            toReturn = True
        return toReturn
