from typing import Any, Counter, Tuple
from numpy.typing import NDArray
from PIL import Image
import numpy as np
from skimage.color import deltaE_ciede2000
from color_utils import ColorUtils

def reducir_imagen(original_matrix: NDArray[np.uint8], nuevo_tamaño: Tuple[int, int])-> NDArray[np.uint8]:
    height, width, rgb = original_matrix.shape
    new_height, new_width = nuevo_tamaño

    # Factor de reducción (debe ser exacto)
    factor_h = height // new_height
    factor_w = width // new_width

    reduced_matrix = np.zeros((new_height, new_width, rgb), dtype=np.uint8)
    # Redimensiona por bloques: reshape + mean
    for i in range(new_height):
        for j in range(new_width):
            reduced_matrix[i,j] = original_matrix[i*factor_h,j*factor_w]

    return reduced_matrix

def eliminar_fondo_por_color(original_matrix: NDArray[np.float64], tolerancia=30):
    height, width, rgb = original_matrix.shape
    # Paso 1: tomamos el color del fondo (esquina superior izquierda por ejemplo)
    fondo_lab = original_matrix[0, 0]  # [R, G, B]

    # Paso 2: calculamos distancia a cada píxel
    for i in range(height):
        for j in range(width):
            pixel_lab = original_matrix[i, j]
            diferencia = ColorUtils.delta_ciede2000(fondo_lab, pixel_lab)
            print(diferencia)
            if diferencia < tolerancia:
                original_matrix[i, j] = [0,0,0]

    return original_matrix

def unify_sub_matrices_color(original_matrix: NDArray[np.float64], div_factor = 32) -> NDArray[np.float64]:
    height, width, rgb = original_matrix.shape
    sub_matrix_height = height // div_factor
    sub_matrix_width = width // div_factor
    for i in range(div_factor):
        for j in range(div_factor):

            counter = Counter()
            for a in range(sub_matrix_height*i,sub_matrix_height*(i+1)):
                for b in range(sub_matrix_width*j,sub_matrix_width*(j+1)):
                    if(tuple(original_matrix[a,b])!=(0,0,0)):
                        print(tuple(original_matrix[a,b]))
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

    for _ in range(1, n):
        # Calcular la mínima distancia de cada color restante al conjunto ya seleccionado
        distances = np.array([
            np.min(deltaE_ciede2000(np.tile(color[None, :], (len(selected), 1)), np.array(selected)))
            for color in remaining
        ])

        # Elegir el color con mayor distancia mínima
        idx_max = np.argmax(distances)
        selected.append(remaining[idx_max])
        remaining = np.delete(remaining, idx_max, axis=0)

    return np.array(selected)

def __closest_color(color: np.ndarray, palette: NDArray[np.float64]) -> NDArray[Any]:
    most_similar_value = 1000
    closest_color = palette[0]
    for i in range(len(palette)):
        delta = deltaE_ciede2000(color,palette[i])
        print(delta)
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

# Abrir la imagen
sword_image = Image.open("resources/pixel_sword_1024x1024.png").convert("RGB")  # Asegura que sea RGB
red_potion_image = Image.open("resources/pocion_roja_ultrarealista.png").convert("RGB")  # Asegura que sea RGB

# Convertir a matriz NumPy
matriz_1024x1024: NDArray[np.uint8] = np.array(sword_image)

matriz_256 = reducir_imagen(matriz_1024x1024, (256, 256))
matriz_128 = reducir_imagen(matriz_1024x1024, (128, 128))

lab_matrix = ColorUtils.transform_matrix_from_rgb_to_lab(matriz_128)

resultado = draw_main_colors(lab_matrix,16)

#resultado = eliminar_fondo_por_color(lab_matrix, tolerancia=20)

resultado = unify_sub_matrices_color(resultado,div_factor=128)
resultado = unify_sub_matrices_color(resultado,div_factor=64)
resultado = blacken_background(resultado)
resultado = fill_image_gaps(resultado,5)
rgb_matrix = ColorUtils.transform_matrix_from_lab_lo_rgb(resultado)
Image.fromarray(rgb_matrix).show()
