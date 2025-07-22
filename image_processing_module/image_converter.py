import math
from typing import Any, Counter, List, Tuple
from numpy.typing import NDArray
from PIL import Image
import numpy as np

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
                    counter[tuple(original_matrix[a,b])]+=1

            most_common = counter.most_common(1)[0][0]
            for a in range(sub_matrix_height*i,sub_matrix_height*(i+1)):
                for b in range(sub_matrix_width*j,sub_matrix_width*(j+1)):
                    original_matrix[a,b] = most_common

    return original_matrix

def show_image(matriz: NDArray[np.uint8]):
    imagen = Image.fromarray(matriz, 'RGB')
    imagen.show()  # o .save("salida.jpg")

def draw_main_colors(original_matrix: NDArray[np.uint8])-> NDArray[np.uint8]:
    height, width, _ = original_matrix.shape
    main_colors = __get_main_colors(original_matrix)
    palette = np.array(main_colors)
    #palette = __reduce_palette_by_distance(palette, len(palette)/4)
    for i in range(height):
        for j in range(width):
            original_matrix[i, j] = __closest_color(original_matrix[i, j], palette)

    return original_matrix

def draw_main_colors_v2(original_matrix: NDArray[np.uint8]) -> NDArray[np.uint8]:
    height, width, _ = original_matrix.shape
    main_colors = __get_main_colors(original_matrix)
    intermediate_colors_palette = __get_intermediate_colors()
    best_representatives_palette = __get_best_representatives(main_colors,intermediate_colors_palette)
    for i in range(height):
        for j in range(width):
            original_matrix[i, j] = __closest_color(original_matrix[i, j], best_representatives_palette)

    return original_matrix

#Retorna el color mas comun
def __get_main_color(original_matrix: NDArray[np.uint8]) -> np.ndarray:
    height, width, _ = original_matrix.shape
    counter = Counter()
    for i in range(height):
        for j in range(width):
            counter[tuple(original_matrix[i,j])]+=1
    return [elemento for elemento, _ in counter.most_common(1)[0]]
    
#Retorna los n elementos mas comunes. N = cantidad de colores // div factor
def __get_main_colors(original_matrix: NDArray[np.uint8], div_factor = 3) -> NDArray[Any]:
    height, width, _ = original_matrix.shape
    counter = Counter()
    for i in range(height):
        for j in range(width):
            counter[tuple(original_matrix[i,j])]+=1
    return [elemento for elemento, _ in counter.most_common(len(counter)//div_factor)]

def __closest_color(color: np.ndarray, palette: NDArray[np.uint8]) -> NDArray[Any]:
    # Calcula la distancia euclidiana entre `color` y cada color de la paleta
    distances = np.linalg.norm(palette - color, axis=1)
    return palette[np.argmin(distances)]

def __reduce_palette_by_distance(palette: NDArray[np.uint8], target_size: int) -> np.ndarray:
    # Iniciar con el color más diferente al promedio
    selected = []
    while len(selected) < target_size:
        average_color = np.mean(palette, axis=0)
        #Color mas diferente al average, medido con distancia euclidiana
        most_different_color_index = np.argmax(np.linalg.norm(palette - average_color, axis=1))
        selected.append(palette[most_different_color_index])
        palette = np.delete(palette, most_different_color_index, axis=0)
    
    return np.array(selected)

def __get_intermediate_colors(n: int = 3) -> NDArray[np.uint8]:
    rgb_constant = 256
    palette = []
    for i in range(n):
        for j in range(n):
            for k in range(n):
                palette.append((
                    math.ceil(rgb_constant * i / n),
                    math.ceil(rgb_constant * j / n),
                    math.ceil(rgb_constant * k / n)
                ))
    return np.array(palette, dtype=np.uint8)

#Retorna los mejores representantes que hay en la lista de colores, de la paleta ingresada
def __get_best_representatives(colors: NDArray[np.uint8], palette: NDArray[np.uint8]) -> NDArray[np.uint8]:
    num_colors = len(palette)
    counter_list: List[Counter[Tuple[int, int, int]]] = [Counter() for _ in range(num_colors)]
    best_representatives: List[Tuple[int, int, int]] = []

    for color in colors:
        index = __closest_color_index(color, palette)
        counter_list[index][color] += 1

    for counter in counter_list:
        most_common_list = counter.most_common(1)
        if len(most_common_list)>0:
            best_representatives.append(most_common_list[0][0])

    return np.array(best_representatives)

def __closest_color_index(color: np.ndarray, palette: NDArray[Any]) -> NDArray[Any]:
    # Calcula la distancia euclidiana entre `color` y cada color de la paleta
    distances = np.linalg.norm(palette - color, axis=1)
    return np.argmin(distances)

def count_unique_colors(rgb_matrix: np.ndarray) -> int:
    # Asegura que la matriz tenga 3 canales (RGB)
    assert rgb_matrix.ndim == 3 and rgb_matrix.shape[2] == 3, "Debe ser una matriz RGB de forma (H, W, 3)"
    
    # Convierte cada pixel a una tupla para poder usarlo como clave
    flat_pixels = rgb_matrix.reshape(-1, 3)
    
    # Usa un set de tuplas (inmutable y hashable)
    unique_colors = set(map(tuple, flat_pixels))
    
    return len(unique_colors)

def fill_image_gaps(original_matrix: NDArray[np.uint8], d: int = 10) -> NDArray[np.uint8]:
    height, width, _ = original_matrix.shape
    for i in range(height):
        black_row = 0
        for j in range(width):
            if ((all(original_matrix[i, j] == np.array([0, 0, 0])))):
                black_row+=1
            else:
                if (black_row<d):
                    original_matrix = __paint_row_segment(original_matrix,original_matrix[i,j],i,black_row,j-1)
                black_row = 0

    for j in range(width):
        black_row = 0
        for i in range(height):
            if ((all(original_matrix[i, j] == np.array([0, 0, 0])))):
                black_row+=1
            else:
                if (black_row<d):
                    original_matrix = __paint_column_segment(original_matrix,original_matrix[i,j],j,black_row,i-1)
                black_row = 0

    return original_matrix

def __paint_row_segment(original_matrix: NDArray[np.uint8], color: np.ndarray, i:int, d: int, end_point: int) -> NDArray[np.uint8]:
    for j in range(d):
        original_matrix[i,end_point-j] = color
    return original_matrix

def __paint_column_segment(original_matrix: NDArray[np.uint8], color: np.ndarray, j:int, d: int, end_point: int) -> NDArray[np.uint8]:
    for i in range(d):
        original_matrix[end_point-i,j] = color
    return original_matrix

# Abrir la imagen
sword_image = Image.open("resources/pixel_sword_1024x1024.png").convert("RGB")  # Asegura que sea RGB
red_potion_image = Image.open("resources/pocion_roja_ultrarealista.png").convert("RGB")  # Asegura que sea RGB

# Convertir a matriz NumPy
matriz_1024x1024: NDArray[np.uint8] = np.array(sword_image)

matriz_256 = reducir_imagen(matriz_1024x1024, (256, 256))

lab_matrix = ColorUtils.transform_matrix_from_rgb_to_lab(matriz_256)

#resultado = draw_main_colors(matriz_256)

#resultado = draw_main_colors_v2(matriz_256)

#resultado = matriz_delta(matriz_256)

#resultado = whiten_matrix(matriz_256)

resultado = eliminar_fondo_por_color(matriz_256, tolerancia=70)

resultado = unify_sub_matrices_color(resultado,div_factor=128)
#resultado = unify_sub_matrices_color(resultado,div_factor=64)
#resultado = unify_sub_matrices_color(resultado,div_factor=32)
resultado = fill_image_gaps(resultado)

Image.fromarray(resultado).show()
