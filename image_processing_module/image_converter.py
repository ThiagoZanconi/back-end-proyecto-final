import math
from typing import Any, Counter, List, Set, Tuple
from matplotlib import pyplot as plt
from numpy.typing import NDArray
from PIL import Image
import numpy as np
from color_utils import ColorUtils
from segment_analyzer_service import SegmentAnalyzerService
from shape_analyzer_service import Segment, ShapeAnalyzerService
from matrix_color_service import MatrixColorService

def reducir_imagen(original_matrix: NDArray[np.uint8], nuevo_tamaño: Tuple[int, int]) -> NDArray[np.uint8]:
    height, width, rgb = original_matrix.shape
    new_height, new_width = nuevo_tamaño

    # Coordenadas de división de bloques sin pérdida/solape
    ys = np.linspace(0, height, new_height + 1, dtype=int)
    xs = np.linspace(0, width, new_width + 1, dtype=int)

    reduced = np.zeros((new_height, new_width, rgb), dtype=np.uint8)

    for i in range(new_height):
        for j in range(new_width):
            y0, y1 = ys[i], ys[i+1]
            x0, x1 = xs[j], xs[j+1]
            block = original_matrix[y0:y1, x0:x1]
            reduced[i, j] = block.mean(axis=(0, 1)).astype(np.uint8)

    return reduced

def unify_sub_matrices_color(original_matrix: NDArray[np.float64], div_factor = 32) -> NDArray[np.float64]:
    height, width, _ = original_matrix.shape
    sub_matrix_height = height // div_factor
    sub_matrix_width = width // div_factor
    for i in range(div_factor):
        for j in range(div_factor):

            counter = Counter()
            for a in range(sub_matrix_height*i,sub_matrix_height*(i+1)):
                for b in range(sub_matrix_width*j,sub_matrix_width*(j+1)):
                    counter[tuple(original_matrix[a,b])]+=1

            most_common = counter.most_common(1)[0][0]
            for a in range(sub_matrix_height*i,sub_matrix_height*(i+1)):
                for b in range(sub_matrix_width*j,sub_matrix_width*(j+1)):
                    original_matrix[a,b] = most_common

    return original_matrix

def blacken_background(original_matrix: NDArray[np.float64]) -> NDArray[np.float64]:
    height, width, rgb = original_matrix.shape
    counter = Counter()
    for i in range(height):
        for j in range(width):
            counter[tuple(original_matrix[i,j])]+=1
    most_common = counter.most_common(1)[0][0]
    for i in range(height):
        for j in range(width):
            if(tuple(original_matrix[i,j]) == most_common):
                original_matrix[i, j] = [0,0,0]
    return original_matrix

def draw_shape(original_matrix: NDArray[np.float64]) -> NDArray[np.float64]:
    height, width, rgb = original_matrix.shape
    shape_matrix = np.zeros((height, width, 3), dtype=np.float64)
    for i in range(1,height-1):
        for j in range(1,width-1):
            if(tuple(original_matrix[i,j])!= (0,0,0) and (tuple(original_matrix[i,j-1]) == (0,0,0) or tuple(original_matrix[i,j+1]) == (0,0,0) or tuple(original_matrix[i-1,j]) == (0,0,0) or tuple(original_matrix[i+1,j]) == (0,0,0))):
                shape_matrix[i,j] = [100,0,0]
            else:
                shape_matrix[i,j] = [0,0,0]
    return shape_matrix

def draw_main_colors(original_matrix: NDArray[np.float64], n = 10) -> NDArray[np.float64]:
    height, width, _ = original_matrix.shape
    unique_colors = np.unique(original_matrix.reshape(-1, 3), axis=0)
    most_diff = __get_most_different_colors(unique_colors, n)
    for i in range(height):
        for j in range(width):
            original_matrix[i, j] = __closest_color(original_matrix[i, j], most_diff)

    return original_matrix

def __get_most_different_colors(lab_colors: NDArray[np.float64], n: int = 10) -> NDArray[np.float64]:
    selected = [lab_colors[0]]  # Empezamos con un color arbitrario (el primero)
    remaining = lab_colors[1:]

    for _ in range(n):
        max_delta = 0
        idx_max = 0
        for i in range(len(remaining)):
            current_delta = min(ColorUtils.delta_ciede2000(remaining[i], s) for s in selected)
            if(current_delta>max_delta):
                max_delta = current_delta
                idx_max = i
                
        selected.append(remaining[idx_max])
        remaining = np.delete(remaining, idx_max, axis=0)

    return np.array(selected)

def __closest_color(color: np.ndarray, palette: NDArray[np.float64]) -> NDArray[Any]:
    most_similar_value = 1000
    closest_color = palette[0]
    for i in range(len(palette)):
        delta = ColorUtils.delta_ciede2000(color,palette[i])
        if(delta < most_similar_value):
            most_similar_value = delta
            closest_color = palette[i]

    return closest_color

def fill_image_gaps(original_matrix: NDArray[np.float64], d: int = 10) -> NDArray[np.float64]:
    height, width, _ = original_matrix.shape
    for i in range(height):
        black_row = 0
        for j in range(width):
            if np.array_equal(original_matrix[i, j], [0.0, 0.0, 0.0]):
                black_row += 1
            else:
                if black_row < d and black_row > 0:
                    original_matrix = __paint_row_segment(original_matrix, original_matrix[i, j], i, black_row, j - 1)
                black_row = 0

    for j in range(width):
        black_row = 0
        for i in range(height):
            if np.array_equal(original_matrix[i, j], [0.0, 0.0, 0.0]):
                black_row += 1
            else:
                if black_row < d and black_row > 0:
                    original_matrix = __paint_column_segment(original_matrix, original_matrix[i, j], j, black_row, i - 1)
                black_row = 0

    return original_matrix

def __paint_row_segment(original_matrix: NDArray[np.float64], color: np.ndarray, i:int, d: int, end_point: int) -> NDArray[np.uint8]:
    for j in range(d):
        original_matrix[i,end_point-j] = color
    return original_matrix

def __paint_column_segment(original_matrix: NDArray[np.float64], color: np.ndarray, j:int, d: int, end_point: int) -> NDArray[np.uint8]:
    for i in range(d):
        original_matrix[end_point-i,j] = color
    return original_matrix

def border_list(path:str)-> List[Tuple[int,int]]:
    sword_image = Image.open(path).convert("RGB")  # Asegura que sea RGB
    # Convertir a matriz NumPy
    matriz_1024x1024: NDArray[np.uint8] = np.array(sword_image)
    matriz_128 = reducir_imagen(matriz_1024x1024, (256, 256))
    lab_matrix = ColorUtils.transform_matrix_from_rgb_to_lab(matriz_128)
    matrix_color_service = MatrixColorService(lab_matrix, delta_threshold = 10)
    return matrix_color_service.border_list()

def graficar_segmentos(segmentos: List[Segment]):
    if not segmentos:
        return
    # Graficar el primer segmento en rojo
    (i1, j1) = segmentos[0].first
    (i2, j2) = segmentos[0].last
    plt.plot([j1, j2], [i1, i2], 'ro-', label='Primer segmento')  # 'r' = rojo
    # Graficar el resto en azul
    for segment in segmentos[1:]:
        (i1, j1) = segment.first
        (i2, j2) = segment.last
        plt.plot([j1, j2], [i1, i2], 'bo-')  # 'b' = azul
    plt.gca().set_aspect('equal')
    plt.gca().invert_yaxis()  # Para que (0,0) esté arriba a la izquierda
    plt.grid(True)
    plt.legend()
    plt.show()

def graficar_formas_por_separado(lista_de_formas: List[List[Segment]]):
    for idx, segmentos in enumerate(lista_de_formas):
        plt.figure()
        x_actual, y_actual = 0, 0

        for segment in segmentos:
            (i1, j1) = segment.first
            (i2, j2) = segment.last
            dx = j2 - j1
            dy = i2 - i1

            x_nuevo = x_actual + dx
            y_nuevo = y_actual + dy

            plt.plot([y_actual, y_nuevo], [x_actual, x_nuevo], 'bo-')
            x_actual, y_actual = x_nuevo, y_nuevo

        plt.gca().set_aspect('equal')
        plt.gca().invert_xaxis()
        plt.grid(True)
        plt.title(f"Forma {idx+1} trasladada desde (0, 0)")
        plt.show()

def graficar_segmentos_origen(segmentos: List[Segment]):
    plt.figure()
    x1, y1 = 0, 0

    for segment in segmentos:
        (i1, j1) = segment.first
        (i2, j2) = segment.last
        dx = j2 - j1
        dy = i2 - i1
        nx1, ny1 = x1 + dx, y1 + dy

        plt.plot([y1, ny1], [x1, nx1], 'bo-')

        x1, y1 = nx1, ny1

    plt.gca().set_aspect('equal')
    plt.gca().invert_xaxis()
    plt.grid(True)
    plt.title(f"Segmentos de (0,0)")
    plt.show()

'''
# Abrir la imagen
sword_image = Image.open("resources/pixel_sword_1024x1024.png").convert("RGB")  # Asegura que sea RGB
# Convertir a matriz NumPy
matriz_1024x1024: NDArray[np.uint8] = np.array(sword_image)
matriz_256 = reducir_imagen(matriz_1024x1024, (256, 256))
matriz_128 = reducir_imagen(matriz_1024x1024, (128, 128))
lab_matrix = ColorUtils.transform_matrix_from_rgb_to_lab(matriz_128)
resultado = draw_main_colors(lab_matrix,14)
resultado = unify_sub_matrices_color(resultado,div_factor=128)
resultado = unify_sub_matrices_color(resultado,div_factor=64)
resultado = unify_sub_matrices_color(resultado,div_factor=32)
resultado = blacken_background(resultado)
resultado = fill_image_gaps(resultado,5)
'''

images = ["resources/swords/pixel_sword_3.png","resources/swords/pixel_sword_4.png", "resources/swords/pixel_sword_2.png", "resources/swords/pixel_sword.avif"]
images_borders = []
for image in images:
    images_borders.append(border_list(image))

shape_analyzer_service = ShapeAnalyzerService(images_borders,20)
segment_analyzer = SegmentAnalyzerService(shape_analyzer_service.shapes_segment_list)

for i in range(8):
    lab_matrix = segment_analyzer.new_matrix_shape(300)
    rgb_matrix = ColorUtils.transform_matrix_from_lab_lo_rgb(lab_matrix)
    Image.fromarray(rgb_matrix).show()
    input("Presiona Enter para mostrar la siguiente imagen...")

#rgb_matrix = ColorUtils.transform_matrix_from_lab_lo_rgb(resultado)
#Image.fromarray(rgb_matrix).show()
