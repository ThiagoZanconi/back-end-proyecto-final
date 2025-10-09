import os
import sys
import requests

API_KEY = os.getenv("DEEPMIND_API_KEY")  # pon tu key en la variable de entorno DEEPMIND_API_KEY

if not API_KEY:
    print("ERROR: define la variable de entorno DEEPMIND_API_KEY con tu API key.")
    sys.exit(1)

# Endpoint para listar modelos (puede variar según la versión; este es un endpoint público de ejemplo).
# Si tu cuenta usa otra ruta (vertex ai / generativelanguage), ajustá la URL según corresponda.
URL = "https://generativelanguage.googleapis.com/v1/models"

headers = {
    "x-goog-api-key": API_KEY,
    "Accept": "application/json",
}

try:
    resp = requests.get(URL, headers=headers, timeout=10)
except requests.RequestException as e:
    print("Error de conexión:", e)
    sys.exit(2)

print("Status code:", resp.status_code)

# Interpretación básica de status codes:
if resp.status_code == 200:
    print("✅ API key válida — respuesta recibida.")
    # Mostrar un resumen de modelos (si vienen)
    try:
        data = resp.json()
        models = data.get("models") or data.get("model") or data  # defensivo
        print("Respuesta (resumen):")
        if isinstance(models, list):
            for m in models[:10]:
                name = m.get("name") if isinstance(m, dict) else str(m)
                print(" -", name)
        else:
            # imprime parte del json para inspección rápida
            print(models if len(str(models)) < 400 else str(models)[:400] + "...")
    except ValueError:
        print("⚠️ Respuesta no JSON o JSON inválido.")
elif resp.status_code in (401, 403):
    print("❌ API key inválida o sin permisos (401/403). Revisa la key y los permisos de la API.")
elif resp.status_code == 404:
    print("⚠️ Endpoint no encontrado (404). ¿La URL es la correcta para tu proyecto/servicio?")
else:
    print("❗Respuesta inesperada:", resp.text[:800])