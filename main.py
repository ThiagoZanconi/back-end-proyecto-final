from contextlib import asynccontextmanager
import os
import shutil
import tempfile
from typing import List
from fastapi import FastAPI, HTTPException
import numpy as np
from image_processing_module.image_processing_service import ImageProcessingService
from request_types.gamma_request import GammaRequest
from routers import ai_assistant

#tmp_dir: str|None = None  # global para guardar la ruta

image_processing_service: ImageProcessingService | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global image_processing_service
    global tmp_dir
    tmp_dir = tempfile.mkdtemp()
    print(f"📂 Carpeta temporal creada: {tmp_dir}")

    # crear instancia del service ya con tmp_dir correcto
    image_processing_service = ImageProcessingService(tmp_dir)

    yield  # <-- Aquí corre la API mientras está viva

    # 🔹 Se ejecuta al apagar la API
    if tmp_dir and os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
        print(f"🗑️ Carpeta temporal eliminada: {tmp_dir}")

def get_image_service() -> ImageProcessingService:
    if image_processing_service is None:
        raise HTTPException(status_code=500, detail="ImageProcessingService no está inicializado")
    return image_processing_service

app = FastAPI(lifespan=lifespan)
app.include_router(ai_assistant.router)

@app.get("/")
def read_root():
    return {"Hello": "World"}

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
