from typing import Any
from numpy.typing import NDArray
from PIL import Image
import numpy as np

def reducir_imagen(original_matrix: NDArray[Any], nuevo_tamaño: tuple[int, int]):
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

def eliminar_fondo_por_color(matriz, tolerancia=30):
    # Paso 1: tomamos el color del fondo (esquina superior izquierda por ejemplo)
    rgb = matriz[0, 0]  # [R, G, B]

    # Paso 2: calculamos distancia a cada píxel
    diferencia = np.linalg.norm(matriz - rgb, axis=2)

    # Paso 3: creamos una máscara de qué píxeles se parecen al fondo
    fondo_mask = diferencia < tolerancia

    # Paso 4: aplicamos la máscara para ponerlos en negro
    matriz_sin_fondo = matriz.copy()
    matriz_sin_fondo[fondo_mask] = [0, 0, 0]

    return matriz_sin_fondo

#Matriz donde en la celda i,j tiene una terna con la diferencia de la matriz i,j y la matriz i+1,j+1
def matriz_delta(original_matrix: NDArray[Any]):
    height, width, rgb = original_matrix.shape
    delta_matrix = np.zeros((height, width, rgb), dtype=np.uint8)
    for i in range(height-1):
        for j in range(width-1):
            delta_matrix[i,j] = np.abs(original_matrix[i,j] - original_matrix[i+1,j+1]).astype(np.uint8)
    return delta_matrix

def show_image(matriz):
    imagen = Image.fromarray(matriz, 'RGB')
    imagen.show()  # o .save("salida.jpg")

# Abrir la imagen
imagen = Image.open("resources/pixel_sword_1024x1024.png").convert("RGB")  # Asegura que sea RGB

# Convertir a matriz NumPy
matriz: NDArray[Any] = np.array(imagen)

matriz_256 = reducir_imagen(matriz, (256, 256))

matriz_d = matriz_delta(matriz_256)

show_image(matriz_d)

#resultado = eliminar_fondo_por_color(matriz_256, tolerancia=20)
#Image.fromarray(resultado).show()
