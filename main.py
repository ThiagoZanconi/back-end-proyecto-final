from contextlib import asynccontextmanager
import os
import shutil
import tempfile
from typing import List, Union
from fastapi import FastAPI
import numpy as np
from image_processing_module.image_processing_service import ImageProcessingService

tmp_dir: str|None = None  # global para guardar la ruta

image_processing_service: ImageProcessingService | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global image_processing_service
    tmp_dir = tempfile.mkdtemp()
    print(f"📂 Carpeta temporal creada: {tmp_dir}")

    # crear instancia del service ya con tmp_dir correcto
    image_processing_service = ImageProcessingService(tmp_dir)

    yield  # <-- Aquí corre la API mientras está viva

    # 🔹 Se ejecuta al apagar la API
    if tmp_dir and os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
        print(f"🗑️ Carpeta temporal eliminada: {tmp_dir}")

app = FastAPI(lifespan=lifespan)

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}

@app.get("/tmp_folder_path/")
def read_item():
    return {"tmp_folder_path": tmp_dir}

@app.get("/most_different_colors/")
def most_different_colors(path: str, n: int = 10, delta_threshold: float = 3):
    colors: List[np.ndarray] = image_processing_service.get_main_different_colors(path, n, delta_threshold)
    colors_list = [c.tolist() for c in colors]
    return {"colors": colors_list}

@app.post("/undo/")
def undo():
    image_processing_service.undo()
    return {"msg": "Undo succesful"}

@app.post("/redo/")
def redo():
    image_processing_service.redo()
    return {"msg": "Redo succesful"}

@app.post("/resize_image/")
def resize_image(path: str, new_h:int, new_w:int):
    image_processing_service.resize_image(path, (new_h, new_w))
    return {"msg": "Image enlarged correctly"}

@app.post("/remove_background/")
def remove_background(path: str):
    image_processing_service.remove_background(path)
    return {"msg": "Fondo removido correctamente"}

@app.post("/extract_border/")
def extract_border(path: str):
    image_processing_service.extract_border(path)
    return {"msg": "Image enlarged correctly"}
