# 🌄 Image editor back-end

Este proyecto utiliza modelos de lenguaje e inteligencia artificial para **predecir tendencias del mercado bursátil** y generar visualizaciones o imágenes relacionadas mediante distintos backends de IA.

---

## 🚀 Uso

### 1. Crear entorno virtual con Python 3.11.0

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

---

## 🧠 Modelos de generación de imágenes

Después de configurar la API key de Hugging Face, puedes instalar los modelos disponibles ejecutando los archivos correspondientes en el módulo:

```bash
cd ai_image_generator_module
```

Modelos soportados:

* `flux1-schnell`
* `sdxl-turbo`

---

## 🤖 Instalación de Ollama

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

**Desarrollado con ❤️ y Python 3.11**
