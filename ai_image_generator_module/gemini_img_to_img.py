from google import genai
from google.genai import types
import os
from google import genai
from PIL import Image
from io import BytesIO

from dotenv import load_dotenv
load_dotenv()

def gemini_image_to_image(prompt: str, output_path: str, image_path: str):
    api_key = os.getenv("DEEPMIND_API_KEY")
    if not api_key:
        raise RuntimeError("Definí la variable DEEPMIND_API_KEY con tu API Key")
    try:
        # Inicializar el cliente (busca la clave API en las variables de entorno)
        client = genai.Client(api_key=api_key)

        # 1. Cargar la imagen usando PIL (Pillow)
        print(f"Cargando imagen desde: {image_path}")
        imagen_entrada = Image.open(image_path)
        
        # 2. Preparar el contenido multimodal: el prompt de texto y la imagen
        contenido_para_gemini = [
            prompt, 
            imagen_entrada
        ]

        # 3. Configuración para solicitar una respuesta que contenga solo la imagen (opcional)
        config = types.GenerateContentConfig(
            response_modalities=[types.Modality.IMAGE],
            # Si quieres texto de respuesta además de la imagen:
            # response_modalities=[types.Modality.TEXT, types.Modality.IMAGE], 
        )

        # 4. Llamar a la API para generar/editar el contenido
        response = client.models.generate_content(
            model="gemini-2.5-flash-image",
            contents=contenido_para_gemini,
            config=config,
        )

        # 5. Procesar la respuesta y guardar la imagen
        imagen_parte = next(
            (part for part in response.candidates[0].content.parts if part.inline_data),
            None
        )

        if imagen_parte and imagen_parte.inline_data:
            # Los datos de la imagen vienen en el formato 'inline_data'
            datos_imagen = imagen_parte.inline_data.data
            
            # Crear y guardar la imagen usando BytesIO y PIL
            imagen_salida = Image.open(BytesIO(datos_imagen))
            imagen_salida.save(output_path)
            
            print(f"\n✅ Imagen editada guardada exitosamente en: {output_path}")
        else:
            print("\n❌ No se recibió una imagen en la respuesta de la API.")

    except FileNotFoundError:
        print(f"Error: No se encontró el archivo de imagen en la ruta: {image_path}")
    except Exception as e:
        print(f"Ocurrió un error al comunicarse con la API de Gemini: {e}")

if __name__ == "__main__":
    gemini_image_to_image("put mafia hats on these dogs", "Images/pixar_style.png", "Images/dogs_2.png")