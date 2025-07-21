from PIL import Image
import matplotlib.pyplot as plt
from numpy.typing import NDArray
import numpy as np
from typing import Tuple

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

# Cargar la imagen
image_path = "resources/pixel_sword_1024x1024.png"
image = Image.open(image_path).convert("RGB")
matriz_1024x1024: NDArray[np.uint8] = np.array(image)
image = reducir_imagen(matriz_1024x1024, (256,256))

# Mostrar imagen
fig, ax = plt.subplots()
ax.imshow(image)
ax.set_title("Haz clic en un píxel")

# Función al hacer clic
def on_click(event):
    if event.inaxes:
        x, y = int(event.xdata), int(event.ydata)
        rgb = image.getpixel((x, y))
        print(f"Coordenadas: ({x}, {y}) - RGB: {rgb}")
        ax.set_title(f"({x}, {y}) - RGB: {rgb}")
        fig.canvas.draw()

# Conectar evento de clic
fig.canvas.mpl_connect('button_press_event', on_click)

plt.show()
