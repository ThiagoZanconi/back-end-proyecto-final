from contextlib import asynccontextmanager
import os
from pathlib import Path
import shutil
import tempfile
from typing import List
from PIL import UnidentifiedImageError
from fastapi import FastAPI, HTTPException
import numpy as np
from image_processing_module.image_processing_service import ImageProcessingService
from request_types.gamma_request import GammaRequest
from routers import ai_assistant

#tmp_dir: str|None = None  # global para guardar la ruta
file_path = Path(__file__).parent.parent / "front-end-proyecto-final" / "image-editor" / "public" 

image_processing_service: ImageProcessingService | None = None
tmp_files: List[str] = []

@asynccontextmanager
async def lifespan(app: FastAPI):
    global image_processing_service
    global tmp_dir
    tmp_dir = tempfile.mkdtemp()
    print(f"📂 Carpeta temporal creada: {tmp_dir}")

    # crear instancia del service ya con tmp_dir correcto
    image_processing_service = ImageProcessingService(file_path)

    yield  # <-- Aquí corre la API mientras está viva

    __delete_tmp_files()
    if tmp_dir and os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
        print(f"🗑️ Carpeta temporal eliminada: {tmp_dir}")

def get_image_service() -> ImageProcessingService:
    if image_processing_service is None:
        raise HTTPException(status_code=500, detail="ImageProcessingService no está inicializado")
    return image_processing_service

app = FastAPI(lifespan=lifespan)
app.include_router(ai_assistant.router)

@app.get("/tmp_folder_path/")
def read_item():
    print("Tmp dir:", tmp_dir)
    return {"tmp_folder_path": f"{tmp_dir}"}

@app.get("/most_different_colors/")
def most_different_colors(path: str, n: int = 10, delta_threshold: float = 3):
    colors: List[np.ndarray] = image_processing_service.get_main_different_colors_rgb(path, n, delta_threshold)
    colors_list = []
    for c in colors:
        colors_list.append([int(c[0]), int(c[1]), int(c[2])])
    return {"colors": colors_list}

@app.post("/resize_image/")
def resize_image(filename: str, new_h:int, new_w:int):
    new_filename = image_processing_service.resize_image(filename, (new_h, new_w))
    __delete_old_file(filename)
    tmp_files.append(new_filename)
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
    tmp_files.append(new_filename)
    return {
        "msg": "Fondo removido correctamente",
        "filename": new_filename
    }

@app.post("/extract_border/")
def extract_border(path: str):
    image_processing_service.extract_border(path)
    return {"msg": "Image enlarged correctly"}

@app.post("/change_gamma_colors/")
def change_gamma_colors(path: str, req: GammaRequest, delta_threshold: float = 3.0):
    if req.color[0] < 0 or req.color[0] > 100:
        raise HTTPException(
            status_code=422,
            detail="El valor L debe estar entre 0 y 100"
        )
    if req.color[1] < -128 or req.color[1] > 128:
        raise HTTPException(
            status_code=422,
            detail="El valor a debe estar entre -128 y 128"
        )
    if req.color[2] < -128 or req.color[2] > 128:
        raise HTTPException(
            status_code=422,
            detail="El valor b debe estar entre -128 y 128"
        )
    image_processing_service.change_gamma_colors(path, req.color, req.delta, delta_threshold)
    return {"msg": "Image enlarged correctly"}

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
    for filename in tmp_files:
        file_path = image_processing_service.path / filename
        try:
            if file_path.exists():
                file_path.unlink()
                print(f"Archivo temporal '{filename}' eliminado.")
            else:
                print(f"El archivo temporal '{filename}' no existe, no se puede eliminar.")
        except Exception as e:
            print(f"Error al eliminar el archivo temporal '{filename}': {str(e)}")