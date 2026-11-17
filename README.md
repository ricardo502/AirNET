# DriveGuard AI — prototipo demostrativo en Python

**Importante:** el programa conserva un modo de **datos simulados** y agrega un modo opcional de cámara local mediante OpenCV. La cámara calcula señales básicas de rostro/ojos; no es un modelo de IA entrenado ni un sistema validado para conducción real. Es exclusivamente académico. No instalar ni usar como dispositivo de seguridad en un vehículo.

## Requisitos

- Python 3.11 o posterior, con `tkinter` disponible (habitualmente incluido en instalaciones estándar de Python para Windows).
- `matplotlib` para las gráficas de eventos: `pip install matplotlib`
- `opencv-contrib-python` para la cámara y respaldo Haar: `pip install opencv-contrib-python`
- `mediapipe` para landmarks faciales, ojos, boca e inclinación: `pip install mediapipe`
- Pantalla de computadora; la cámara e Internet no son obligatorios para el modo simulado.

## Ejecutar

1. Descarga y descomprime `DriveGuardAI_Demo.zip`.
2. Abre una terminal en la carpeta `DriveGuardAI_Demo`.
3. Ejecuta:

```bash
python app.py
```

En algunos equipos Windows, se puede usar `py app.py`.

## Acceso local

Al abrir el programa aparece una ventana de acceso creada en `login.py`. En el primer uso, escribe un usuario y una contraseña de al menos ocho caracteres y pulsa **Crear cuenta**. Se abre el panel principal. En los siguientes usos, introduce esos mismos datos y pulsa **Entrar** o presiona Enter desde el campo de contraseña.

Se admite una sola cuenta local, guardada en `data/cuenta.json` con salt aleatorio y hash PBKDF2-HMAC-SHA256, no con la contraseña en texto plano. El archivo está excluido de Git. El modo simulado no requiere paquetes externos; el modo cámara requiere `opencv-python`. No hay recuperación de contraseña ni gestión de múltiples usuarios.

Este acceso es básico para la demostración: no protege el CSV frente a personas con acceso al equipo ni sustituye los permisos del sistema operativo. Los manuales Word generados antes de este cambio todavía no incluyen el login.

## Cómo hacer la demostración

1. Desde **Cámara de la laptop**, prueba la captura local. La cámara genera señales compatibles con `core.py`.
2. Con el estado en `PRECAUCIÓN` o `ALERTA ALTA`, abre **Plan de acción (búsqueda IA)** para ver la secuencia de acciones sugeridas calculada con BFS, DFS, UCS, Greedy o A*.
3. Para probar escenarios sintéticos, abre **Modo simulación** en la barra lateral.
4. En ese apartado puedes seleccionar `Conducción normal`, `Cansancio leve`, `Somnolencia alta` o `Rostro no visible`.
5. Desde **Modo simulación** también puedes iniciar la demostración automática y registrar lecturas simuladas.

## Búsqueda en IA (busqueda.py)

El proyecto aplica los temas de búsqueda sobre un grafo de acciones correctivas:

- **Problema de búsqueda** `P=(S,A,s0,G,C)`: estados = niveles de riesgo (`PRECAUCIÓN`, `ALERTA ALTA`), acciones = correcciones con costo en puntos, estado inicial = nivel actual, objetivo = `NORMAL`.
- **BFS / DFS**: encuentran una ruta de acciones sin usar costos ni heurística.
- **UCS**: ruta de menor costo acumulado `g(n)`.
- **Heurística** `h(n)`: puntos mínimos restantes estimados; admisible sobre el grafo.
- **Greedy**: elige solo por `h(n)` — puede dar una ruta más cara (4) que A* (3).
- **A***: `f(n) = g(n) + h(n)`, encuentra la ruta de menor costo.
- **Hill Climbing**: ajusta un umbral (`blinks_caution_per_minute`) para reducir errores en lecturas etiquetadas.
- **Algoritmo genético**: evoluciona los seis umbrales con torneo, cruza, mutación y semilla fija.

Costos, heurística y lecturas etiquetadas son ilustrativos para la demostración; no corresponden a tiempos reales ni a eficacia validada de las acciones.

## Archivos

```text
DriveGuardAI_Demo/
├── app.py              # Interfaz de escritorio (Tkinter)
├── login.py            # Acceso local con cuenta y contraseña
├── core.py             # Datos, reglas de evaluación y registro CSV
├── busqueda.py         # BFS, DFS, UCS, Greedy, A*, Hill Climbing y genético
├── camera.py           # Captura local y señales básicas con OpenCV
├── busqueda_ui.py      # Ventana "Plan de acción" con los algoritmos
├── graficas.py         # Ventana de gráficas matplotlib sobre eventos.csv
├── requirements.txt    # Sin dependencias externas en el modo simulado
├── tests/
│   └── test_core.py    # Pruebas automatizadas
├── data/
│   └── .gitkeep        # eventos.csv se crea durante la ejecución
└── README.md
```

## Pruebas

```bash
python -m unittest discover -s tests -v
```

## Datos registrados

El archivo `data/eventos.csv` se crea automáticamente cuando cambia el estado a `PRECAUCIÓN`, `ALERTA ALTA` o `SIN ROSTRO`, o cuando el usuario selecciona **Registrar lectura en CSV**. No se guarda ningún video, imagen ni dato real de personas.

## Reglas ilustrativas (NO validadas)

- Ojos cerrados ≥ 1.2 s: +2 puntos.
- Ojos cerrados ≥ 2.5 s: +4 puntos, en lugar de los +2 anteriores.
- Parpadeos ≥ 28 por minuto: +1.
- Bostezos ≥ 3 por minuto: +2.
- Inclinación absoluta ≥ 25°: +2.
- 0–1 puntos: `NORMAL`; 2–3: `PRECAUCIÓN`; ≥ 4: `ALERTA ALTA`.
- Rostro no visible: `SIN ROSTRO`; no se infiere fatiga.

Estos valores fueron elegidos para facilitar la demostración y no deben describirse como umbrales médicos ni como resultado de entrenamiento o validación.

## Alcance de la cámara local

El archivo `camera.py` usa MediaPipe Face Mesh para obtener landmarks de ojos y boca, calcular EAR/MAR, contar parpadeos y bostezos, y estimar la inclinación. OpenCV y Haar quedan como respaldo. Esta etapa no sustituye calibración por persona, manejo de iluminación y oclusiones, pruebas reproducibles ni aviso de privacidad. La tecnología permite extraer señales, **no** identifica la somnolencia de forma infalible.

Referencia: https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker/python
