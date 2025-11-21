# 🇬🇧 English

# 🌄 IMAGEIO - AI IMAGE EDITOR Back-End

This repository contains the back-end of our image generation and editing application.
The front-end can be found here: https://github.com/ThiagoZanconi/front-end-proyecto-final
This module consists of a Python API implemented with FastAPI, which integrates three other submodules (algorithmic, text AI, image AI). These components implement the endpoints used by the front-end to provide the logic behind the different functions available to the user.

---

## 🛠️ Installation

### 1. Create a virtual environment using Python 3.11.0 or 3.12.0

```bash
python -m venv .venv
source .venv/bin/activate    # On Linux / macOS
.venv\Scripts\activate       # On Windows
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

Create a .env file in the project root directory with one of the following keys:

```env
# For Hugging Face
API_KEY=your_hugging_face_api_key

# Or for Gemini
DEEPMIND_API_KEY=your_gemini_api_key
```
  
* Note: If you only want to use Google’s cloud-based image generation through DEEPMIND, you may skip the next step and continue with step 5.
---

## 🧠 4. Local Image Generation Models

After configuring your Hugging Face API key, navigate to the folder:

```bash
cd ai_image_generator_module
```

Then install each model you need by executing the following .py files:

* `flux1-schnell_install.py`
* `sdxl-turbo_install.py`

Currently supported local models:

* [`flux1-schnell`](https://huggingface.co/black-forest-labs/FLUX.1-schnell)
* [`sdxl-turbo`](https://huggingface.co/stabilityai/sdxl-turbo)

---

## 🤖 5. Installing Ollama

1. Download and install [Ollama](https://ollama.com/download).
2. Install one of the supported models directly via Ollama:

```bash
ollama run gpt-oss:20b
```

or

```bash
ollama run deepseek-r1:8b
```

*(You only need to run these once; Ollama will download the model automatically.)*

---
## 🚀 Launch

Once everything is installed, go to the root folder of the project and run the program using the following command:

```bash
fastapi dev main.py
```

---

## ⚙️ Important Considerations

If you change the installation directory of your Ollama models, **you must update the corresponding environment variable** so the program can locate them correctly.

### 🪟 On Windows:

```bash
setx OLLAMA_MODELS "D:\.Ollama\models" /M
```

### 🐧 On Linux:

```bash
export OLLAMA_MODELS="D:\.Ollama\models"
```

## CUDA + PyTorch

If you encounter issues using CUDA cores, or get PyTorch-related errors, visit the following link and download the stable PyTorch version compatible with your system:

https://pytorch.org/get-started/locally/  

---

# 🇪🇸 Español

# 🌄 IMAGEIO - AI IMAGE EDITOR Back-End

Este repositorio contiene el back-end de nuestra aplicacion de generación y edición de imágenes. El front-end se puede encontrar aqui: https://github.com/ThiagoZanconi/front-end-proyecto-final  
  
Este módulo consiste de una API Python, implementada con FastAPI, que integra otros tres submódulos (algorítmico, IA texto, IA imágenes) para implementar los endpoints que utiliza el front-end para proveer la lógica de las distintas funciones disponibles para el usuario en la interfaz.

---

## 🛠️ Instalación

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

## 🚀 Ejecutar

Una vez que todo esta instalado, navega hasta la carpeta raíz del proyecto y ejecuta el programa usando el siguiente comando:

```bash
fastapi dev main.py
```

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
  
**Developed by [Juan Buñes](https://github.com/JuanBunes) & [Thiago Zanconi](https://github.com/ThiagoZanconi)**
