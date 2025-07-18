from typing import Any
from numpy.typing import NDArray
from PIL import Image
import numpy as np
import heapq
import cv2

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
def matriz_delta(original_matrix: NDArray[Any], threshold = 25):
    height, width, rgb = original_matrix.shape
    delta_matrix = np.zeros((height, width, rgb), dtype=np.uint8)
    for i in range(1,height-1):
        for j in range(1,width-1):
            delta_matrix[i,j] = (255,255,255)
            for a in range(i-1,i+1):
                for b in range (j-1,j+1):
                    new_rgb = np.abs(original_matrix[i,j] - original_matrix[a,b]).astype(np.uint8)
                    if(new_rgb[0]+new_rgb[1]+new_rgb[2]>threshold):
                        delta_matrix[i,j] = (0,0,0)
    return delta_matrix

def border_detection(original_matrix):
    # Convertimos a escala de grises primero
    gray = cv2.cvtColor(original_matrix, cv2.COLOR_RGB2GRAY)

    # Aplicar filtro Sobel en x e y
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    # Magnitud del gradiente
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    magnitude = np.clip(magnitude, 0, 255).astype(np.uint8)

    # Si querés volver a RGB
    return cv2.cvtColor(magnitude, cv2.COLOR_GRAY2RGB)

def show_image(matriz):
    imagen = Image.fromarray(matriz, 'RGB')
    imagen.show()  # o .save("salida.jpg")

def whiten_matrix(original_matrix):
    block_size = 32
    for a in range(8):
        for b in range(8):
            white_pixels = 0
            for i in range(a*block_size,(a+1)*block_size):
                for j in range(b*block_size,(b+1)*block_size):
                    if(original_matrix[i,j,0] != 0):
                        white_pixels+=1
            if(white_pixels>160):
                for i in range(a*block_size,(a+1)*block_size):
                    for j in range(b*block_size,(b+1)*block_size):
                        original_matrix[i,j] = (255,255,255)

    return original_matrix

def draw_main_colors(original_matrix: NDArray[Any]):
    print("Use euclidian distances from one colour to the main ones")

# Abrir la imagen
imagen = Image.open("resources/pixel_sword_1024x1024.png").convert("RGB")  # Asegura que sea RGB

# Convertir a matriz NumPy
matriz: NDArray[Any] = np.array(imagen)

matriz_256 = reducir_imagen(matriz, (256, 256))

matriz_d = matriz_delta(matriz_256)

whitened = whiten_matrix(matriz_d)

show_image(whitened)

#resultado = eliminar_fondo_por_color(matriz_256, tolerancia=20)
#Image.fromarray(resultado).show()
