import uuid
from numpy.typing import NDArray
from pathlib import Path
from PIL import Image
import numpy as np
from typing import List, Tuple
from image_processing_module.color_utils import ColorUtils
from image_processing_module.matrix_color_service import MatrixColorService

class ImageProcessingService:
    def __init__(self, path: Path):
        self.path = path

    def remove_background(self, filename: str, delta_threshold = 3) -> str:
        matrix_color_service = self.__instanciate_matrix_color_service(filename, delta_threshold = delta_threshold)
        lab_matrix = matrix_color_service.remover_fondo()
        rgb_matrix = ColorUtils.transform_matrix_from_lab_lo_rgb(lab_matrix)
        return self.__save_image(rgb_matrix)

    def resize_image(self, filename: str, nuevo_tamaño: Tuple[int, int]) -> str:
        rgb_matrix = self.__get_rgb_matrix(self.path / filename)
        reduced = self.__resize_image(rgb_matrix, nuevo_tamaño)
        return self.__save_image(reduced)

    def extract_border(self, filename: str) -> str:
        matrix_color_service = self.__instanciate_matrix_color_service(filename)
        lab_matrix_border = matrix_color_service.border()
        rgb_matrix = ColorUtils.transform_matrix_from_lab_lo_rgb(lab_matrix_border)
        return self.__save_image(rgb_matrix)

    def change_gamma_colors(self, filename: str, color: List[float], new_color: List[float], delta_threshold: float = 3.0) -> str:
        matrix_color_service = self.__instanciate_matrix_color_service(filename, delta_threshold=delta_threshold)
        lab_matrix = matrix_color_service.change_gamma_colors(color, new_color, delta_threshold)
        rgb_matrix = ColorUtils.transform_matrix_from_lab_lo_rgb(lab_matrix)
        return self.__save_image(rgb_matrix)

    def get_main_different_colors_rgb(self, filename: str, n=10, delta_threshold: float = 3.0) -> List[np.uint8]:
        matrix_color_service = self.__instanciate_matrix_color_service(filename, delta_threshold = delta_threshold)
        color_list:List[np.uint8] = matrix_color_service.get_main_different_colors(n, delta_threshold)
        return ColorUtils.lab_color_list_to_rgb(color_list)
    
    def get_color(self, filename: str, x: int, y: int) -> List[int]:
        rgb_matrix = self.__get_rgb_matrix(self.path / filename)
        height, width, rgb = rgb_matrix.shape
        if x < 0 or x >= width or y < 0 or y >= height:
            raise ValueError("Las coordenadas (x, y) están fuera de los límites de la imagen.")
        color = rgb_matrix[y, x]
        return [int(color[0]), int(color[1]), int(color[2])]

    def __instanciate_matrix_color_service(self, filename: str, delta_threshold = 3) -> MatrixColorService:
        rgb_matrix = self.__get_rgb_matrix(self.path / filename)
        lab_matrix = ColorUtils.transform_matrix_from_rgb_to_lab(rgb_matrix)
        return MatrixColorService(lab_matrix, delta_threshold = delta_threshold)

    def __get_rgb_matrix(self, path: Path) -> NDArray[np.uint8]:
        image = Image.open(path).convert("RGB")
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
    
    def __save_image(self, rgb_matrix: NDArray[np.uint8]) -> str:
        imagen = Image.fromarray(rgb_matrix, 'RGB')
        filename = f"{uuid.uuid4()}.png"
        imagen.save(self.path / filename)
        return filename