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
        #matriz_128 = self.__reduce_image(rgb_matrix, (128, 128))
        lab_matrix = ColorUtils.transform_matrix_from_rgb_to_lab(rgb_matrix)
        matrix_color_service = MatrixColorService(lab_matrix, delta_threshold = 6)
        lab_matrix = matrix_color_service.remover_fondo()
        rgb_matrix = ColorUtils.transform_matrix_from_lab_lo_rgb(lab_matrix)
        self.__save_image(rgb_matrix)

    def __get_rgb_matrix(self, path:str) -> NDArray[np.uint8]:
        image = Image.open(path).convert("RGB")  # Asegura que sea RGB
        return np.array(image)

    def __reduce_image(self, original_matrix: NDArray[np.uint8], nuevo_tamaño: Tuple[int, int]) -> NDArray[np.uint8]:
        height, width, rgb = original_matrix.shape
        new_height, new_width = nuevo_tamaño
        reduced = np.zeros((new_height, new_width, rgb), dtype=np.uint8)

        factor_h = height // new_height
        resto_h = (height % new_height) // 2
        factor_w = width // new_width
        resto_w = (width % new_width) // 2

        for i in range(new_height):
            for j in range(new_width):
                reduced[i, j] = original_matrix[i*factor_h + resto_h,j*factor_w + resto_w]
        return reduced
    
    def __save_image(self, rgb_matrix: NDArray[np.uint8]):
        imagen = Image.fromarray(rgb_matrix, 'RGB')
        #ruta = os.path.join(self.tmp_path, "imagen.png")
        print("Tmp path: ",self.tmp_path)
        ruta = self.tmp_path+"/imagen.png"
        imagen.save(ruta)