from contextlib import asynccontextmanager
from pathlib import Path
import shutil
from typing import List
from PIL import UnidentifiedImageError
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
import numpy as np
from image_processing_module.image_processing_service import ImageProcessingService
from request_types.gamma_request import GammaRequest
from routers import ai_assistant
import subprocess

#tmp_dir: str|None = None  # global para guardar la ruta
file_path = Path(__file__).parent.parent / "front-end-proyecto-final" / "image-editor" / "public" 
image_processing_service: ImageProcessingService | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_dotenv()
    global image_processing_service
    image_processing_service = ImageProcessingService(file_path)
    subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL)
    yield  # <-- Aquí corre la API mientras está viva
    __delete_tmp_files()

def get_image_service() -> ImageProcessingService:
    if image_processing_service is None:
        raise HTTPException(status_code=500, detail="ImageProcessingService no está inicializado")
    return image_processing_service

app = FastAPI(lifespan=lifespan)
app.include_router(ai_assistant.router)

@app.get("/most_different_colors/")
def most_different_colors(filename: str, n: int = 10, delta_threshold: float = 3):
    colors: List[np.ndarray] = image_processing_service.get_main_different_colors_rgb(filename, n, delta_threshold)
    colors_list = []
    for c in colors:
        colors_list.append([int(c[0]), int(c[1]), int(c[2])])
    return {"colors": colors_list}

@app.post("/resize_image/")
def resize_image(filename: str, new_h:int, new_w:int):
    new_filename = image_processing_service.resize_image(filename, (new_h, new_w))
    __delete_old_file(filename)
    return {"msg": "Image enlarged correctly",
        "filename": new_filename}

@app.post("/remove_background/")
def remove_background(filename: str, delta_threshold: int = 3):
    print("Received filename:", filename)
    try:
        new_filename = image_processing_service.remove_background(filename, delta_threshold=delta_threshold)

    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"No se encontró el archivo '{filename}'.")

    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail=f"El archivo '{filename}' no es una imagen válida o está dañado.")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error inesperado al procesar la imagen: {str(e)}")
    
    __delete_old_file(filename)
    return {
        "msg": "Fondo removido correctamente",
        "filename": new_filename
    }

@app.post("/extract_border/")
def extract_border(filename: str):
    new_filename = image_processing_service.extract_border(filename)
    __delete_old_file(filename)
    return {
        "msg": "Border extracted correctly",
        "filename": new_filename
    }

@app.post("/change_gamma_colors/")
def change_gamma_colors(filename: str, req: GammaRequest, delta_threshold: float = 3.0):
    for i in range(3):
        if req.new_color[i] < 0 or req.new_color[i] > 255:
            raise HTTPException(
                status_code=422,
                detail="Los valores deben estar entre 0 y 255"
            )
    new_filename = image_processing_service.change_gamma_colors(filename, req.color, req.new_color, delta_threshold)
    __delete_old_file(filename)
    return {
        "msg": "Gamma colors changed correctly",
        "filename": new_filename
    }

@app.get("/get_color/")
def get_color(filename: str, x: int, y: int):
    try:
        color = image_processing_service.get_color(filename, x, y)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    return {
        "color": [int(color[0]), int(color[1]), int(color[2])]
    }

def __delete_old_file(filename: str):
    file_path = image_processing_service.path / filename
    try:
        if file_path.exists():
            file_path.unlink()
            print(f"Archivo antiguo '{filename}' eliminado.")
        else:
            print(f"El archivo '{filename}' no existe, no se puede eliminar.")
    except Exception as e:
        print(f"Error al eliminar el archivo '{filename}': {str(e)}")

def __delete_tmp_files():
    folder_path = image_processing_service.path

    try:
        if folder_path.exists() and folder_path.is_dir():
            # Iterar sobre todos los elementos dentro de la carpeta
            for item in folder_path.iterdir():
                try:
                    if item.is_file() or item.is_symlink():
                        item.unlink()
                    elif item.is_dir():
                        shutil.rmtree(item)
                except Exception as e:
                    print(f"⚠️ No se pudo eliminar '{item}': {e}")

            print(f"✅ Se eliminaron todos los archivos temporales en '{folder_path}'.")
        else:
            print(f"⚠️ La ruta '{folder_path}' no existe o no es un directorio.")
    except Exception as e:
        print(f"❌ Error al limpiar archivos temporales en '{folder_path}': {e}")