import numpy as np
from numpy.typing import NDArray
from typing import Tuple
from PIL import Image
from skimage.color import rgb2lab
import heapq

def reducir_imagen(original_matrix: NDArray[np.uint8], nuevo_tamaño: Tuple[int, int]) -> NDArray[np.uint8]:
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

def draw_color_separation_lines(original_matrix: NDArray[np.uint8], n:int = 6) -> NDArray[np.uint8]:
    height, width, rgb = original_matrix.shape

    delta_rows_pq = []
    delta_columns_pq = []

    #Compare rows
    for i in range(height-1):
        delta_sum: float = 0
        for j in range(width):
            delta_sum += cie2000_delta(original_matrix[i,j],original_matrix[i+1,j]) 

        heapq.heappush(delta_rows_pq, (-delta_sum, i))

    #Compare columns
    for j in range(width-1):
        delta_sum: float = 0
        for i in range(height):
            delta_sum += cie2000_delta(original_matrix[i,j],original_matrix[i,j+1]) 

        heapq.heappush(delta_columns_pq, (-delta_sum, j))

    #Draw row line:
    for index in range(n):
        priority , i = heapq.heappop(delta_rows_pq)
        for j in range(width):
            original_matrix[i,j] = [0,0,0]

    #Draw column line:
    for index in range(n):
        priority , j = heapq.heappop(delta_columns_pq)
        for i in range(height):
            original_matrix[i,j] = [0,0,0]

    return original_matrix

def cie2000_delta(rgb1: np.ndarray, rgb2: np.ndarray) -> float:
    # Asegurarse de que los valores están en el rango 0–1
    rgb1 = np.array(rgb1, dtype=np.float64).reshape(1, 1, 3) / 255.0
    rgb2 = np.array(rgb2, dtype=np.float64).reshape(1, 1, 3) / 255.0

    lab1 = rgb2lab(rgb1)[0, 0]
    lab2 = rgb2lab(rgb2)[0, 0]

    return delta_e_ciede2000(lab1, lab2)

def delta_e_ciede2000(lab1, lab2):
    # Extraer valores
    L1, a1, b1 = lab1
    L2, a2, b2 = lab2

    avg_L = (L1 + L2) / 2.0
    C1 = np.sqrt(a1**2 + b1**2)
    C2 = np.sqrt(a2**2 + b2**2)
    avg_C = (C1 + C2) / 2.0

    G = 0.5 * (1 - np.sqrt((avg_C**7) / (avg_C**7 + 25**7)))
    a1p = (1 + G) * a1
    a2p = (1 + G) * a2

    C1p = np.sqrt(a1p**2 + b1**2)
    C2p = np.sqrt(a2p**2 + b2**2)
    avg_Cp = (C1p + C2p) / 2.0

    h1p = np.degrees(np.arctan2(b1, a1p)) % 360
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360

    deltahp = 0
    if C1p * C2p != 0:
        if abs(h2p - h1p) <= 180:
            deltahp = h2p - h1p
        elif h2p <= h1p:
            deltahp = h2p - h1p + 360
        else:
            deltahp = h2p - h1p - 360

    deltaLp = L2 - L1
    deltaCp = C2p - C1p
    deltaHp = 2 * np.sqrt(C1p * C2p) * np.sin(np.radians(deltahp / 2))

    avg_Hp = 0
    if C1p * C2p != 0:
        if abs(h1p - h2p) > 180:
            avg_Hp = (h1p + h2p + 360) / 2.0
        else:
            avg_Hp = (h1p + h2p) / 2.0
    else:
        avg_Hp = h1p + h2p

    T = 1 - 0.17 * np.cos(np.radians(avg_Hp - 30)) + \
        0.24 * np.cos(np.radians(2 * avg_Hp)) + \
        0.32 * np.cos(np.radians(3 * avg_Hp + 6)) - \
        0.20 * np.cos(np.radians(4 * avg_Hp - 63))

    delta_theta = 30 * np.exp(-((avg_Hp - 275) / 25) ** 2)
    Rc = 2 * np.sqrt((avg_Cp ** 7) / (avg_Cp ** 7 + 25 ** 7))
    Sl = 1 + ((0.015 * ((avg_L - 50) ** 2)) / np.sqrt(20 + ((avg_L - 50) ** 2)))
    Sc = 1 + 0.045 * avg_Cp
    Sh = 1 + 0.015 * avg_Cp * T
    Rt = -np.sin(np.radians(2 * delta_theta)) * Rc

    deltaE = np.sqrt(
        (deltaLp / Sl) ** 2 +
        (deltaCp / Sc) ** 2 +
        (deltaHp / Sh) ** 2 +
        Rt * (deltaCp / Sc) * (deltaHp / Sh)
    )
    return deltaE

sword_image = Image.open("resources/pixel_sword_1024x1024.png").convert("RGB")  # Asegura que sea RGB
# Convertir a matriz NumPy
matriz_1024x1024: NDArray[np.uint8] = np.array(sword_image)

matriz_256 = reducir_imagen(matriz_1024x1024, (256, 256))

resultado = draw_color_separation_lines(matriz_256)
Image.fromarray(resultado).show()
