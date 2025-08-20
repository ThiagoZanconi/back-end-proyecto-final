from numpy.typing import NDArray
from PIL import Image
import numpy as np
from typing import List, Tuple
from image_processing_module.color_utils import ColorUtils
from image_processing_module.matrix_color_service import MatrixColorService

class ImageProcessingService:
    def __init__(self, tmp_path:str):
        self.tmp_path = tmp_path
        self.undo_stack: List[str] = []
        self.redo_stack: List[str] = []
        self.current:str = ""

    def do(self, action):
        self.undo_stack.append(action)
        # Si se hace una acción nueva, se limpia el redo_stack
        self.redo_stack.clear()

    def undo(self):
        if not self.undo_stack:
            return None
        
        action = self.undo_stack.pop()
        self.redo_stack.append(action)
        self.current = action
        return action

    def redo(self):
        if not self.redo_stack:
            return None
        
        action = self.redo_stack.pop()
        self.undo_stack.append(action)
        self.current = action
        return action

    def remove_background(self, path:str):
        rgb_matrix = self.__get_rgb_matrix(path)
        lab_matrix = ColorUtils.transform_matrix_from_rgb_to_lab(rgb_matrix)
        matrix_color_service = MatrixColorService(lab_matrix, delta_threshold = 6)
        lab_matrix = matrix_color_service.remover_fondo()
        rgb_matrix = ColorUtils.transform_matrix_from_lab_lo_rgb(lab_matrix)
        self.__save_image(rgb_matrix)

    def resize_image(self, path:str, nuevo_tamaño: Tuple[int, int]) -> NDArray[np.uint8]:
        rgb_matrix = self.__get_rgb_matrix(path)
        reduced = self.__resize_image(rgb_matrix, nuevo_tamaño)
        self.__save_image(reduced)

    def __get_rgb_matrix(self, path:str) -> NDArray[np.uint8]:
        image = Image.open(path).convert("RGB")  # Asegura que sea RGB
        return np.array(image)
    
    def __resize_image(self, original_matrix: NDArray[np.uint8], nuevo_tamaño: Tuple[int, int]) -> NDArray[np.uint8]:
        height, width, rgb = original_matrix.shape
        new_height, new_width = nuevo_tamaño
        resampled = np.zeros((new_height, new_width, rgb), dtype=np.uint8)

        scale_h = height / new_height
        scale_w = width / new_width

        for i in range(new_height):
            orig_i = int(i * scale_h)
            if orig_i >= height:  # protección borde
                orig_i = height - 1
            for j in range(new_width):
                orig_j = int(j * scale_w)
                if orig_j >= width:  # protección borde
                    orig_j = width - 1
                resampled[i, j] = original_matrix[orig_i, orig_j]

        return resampled
    
    def __save_image(self, rgb_matrix: NDArray[np.uint8]):
        imagen = Image.fromarray(rgb_matrix, 'RGB')
        print("Tmp path: ",self.tmp_path)
        ruta = self.tmp_path+"/imagen.png"
        imagen.save(ruta)