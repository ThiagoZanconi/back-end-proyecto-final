# 🌄 Image editor back-end

Este repositorio contiene el back-end de nuestra aplicacion de generación y edición de imágenes. El front-end se puede encontrar aqui: https://github.com/ThiagoZanconi/front-end-proyecto-final  
  
Este módulo consiste de una API Python, implementada con FastAPI, que integra otros tres submódulos (algorítmico, IA texto, IA imágenes) para implementar los endpoints que utiliza el front-end para proveer la lógica de las distintas funciones disponibles para el usuario en la interfaz.

---

## 🚀 Uso

### 1. Crear entorno virtual con Python 3.11.0 o 3.12.0

```bash
python -m venv .venv
source .venv/bin/activate    # En Linux / macOS
.venv\Scripts\activate       # En Windows
```

### 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 3. Configurar variables de entorno

Crea un archivo `.env` en el directorio raíz con una de las siguientes claves:

```env
# Para Hugging Face
API_KEY=tu_api_key_de_hugging_face

# O para Gemini
DEEPMIND_API_KEY=tu_api_key_de_gemini
```
  
* Nota: Si solo deseas utilizar la generacion de imagenes de Google en la nube a traves de DEEPMIND, puedes saltear el siguiente paso y seguir con el paso 5.
---

## 🧠 4. Modelos de generación de imágenes locales

Después de configurar la API key de Hugging Face, navega a la carpeta:

```bash
cd ai_image_generator_module
```

Y luego instala cada uno de los modelos que necesites, ejecutando los siguientes archivos .py:

* `flux1-schnell_install.py`
* `sdxl-turbo_install.py`

Modelos locales actualmente soportados:

* [`flux1-schnell`](https://huggingface.co/black-forest-labs/FLUX.1-schnell)
* [`sdxl-turbo`](https://huggingface.co/stabilityai/sdxl-turbo)

---

## 🤖 5. Instalación de Ollama

1. Descarga e instala [Ollama](https://ollama.com/download).
2. Instala uno de los modelos soportados directamente desde Ollama:

```bash
ollama run gpt-oss:20b
```

o

```bash
ollama run deepseek-r1:8b
```

*(Solo es necesario ejecutarlos una vez; Ollama descargará el modelo automáticamente.)*

---

## ⚙️ Consideraciones importantes

Si cambias el directorio de instalación de los modelos de Ollama, **debes actualizar la variable de entorno** para que el programa los encuentre correctamente.

### 🪟 En Windows:

```bash
setx OLLAMA_MODELS "D:\.Ollama\models" /M
```

### 🐧 En Linux:

```bash
export OLLAMA_MODELS="D:\.Ollama\models"
```

## CUDA + PyTorch

Si tienes problemas al utilizar los CUDA cores, o recibes errores de PyTorch, dirigete a este link y descarga la version estable de PyTorch que se corresponde con tu equipo:

https://pytorch.org/get-started/locally/  
  
**Desarrollado con ❤️ y Python 3.11**
