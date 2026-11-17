"""Captura de cámara para la demostración local de DriveGuard AI.

Esta integración usa OpenCV para obtener señales aproximadas de la cámara:
rostro visible, ausencia de ojos detectados y duración de ojos cerrados.
No es un modelo de IA entrenado ni un sistema validado para conducción real.
"""
from __future__ import annotations

import base64
import math
import time
import tkinter as tk
from tkinter import messagebox
from pathlib import Path

import cv2

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
except (ImportError, AttributeError):
    mp = None
    mp_python = None
    mp_vision = None

from core import Signals, evaluate


class CameraWindow(tk.Toplevel):
    """Ventana educativa que conecta cámara, evaluación y nivel de riesgo."""

    def __init__(self, master: tk.Misc, on_assessment=None):
        super().__init__(master)
        self.title("DriveGuard AI | Cámara de laptop")
        self.update_idletasks()
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        window_width = min(1000, screen_width - 40)
        window_height = min(680, screen_height - 100)
        self.geometry(f"{window_width}x{window_height}")
        self.minsize(760, 560)
        self.configure(bg="#E9EEF5")
        self.protocol("WM_DELETE_WINDOW", self.close)

        self.video, self.first_frame, self.camera_index = self._open_camera()
        if self.video is None:
            messagebox.showerror(
                "Cámara no disponible",
                "OpenCV no pudo recibir imagen de la cámara.\n\n"
                "Verifica que la cámara no esté siendo usada por Teams, Zoom, "
                "el navegador u otra aplicación y permite el acceso en "
                "Configuración de Windows > Privacidad y seguridad > Cámara.",
                parent=self,
            )
            self.destroy()
            return

        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.eye_cascades = []
        for filename in (
            "haarcascade_eye_tree_eyeglasses.xml",
            "haarcascade_eye.xml",
        ):
            cascade = cv2.CascadeClassifier(cv2.data.haarcascades + filename)
            if not cascade.empty():
                self.eye_cascades.append(cascade)
        self.detector_ready = (
            not self.face_cascade.empty() and bool(self.eye_cascades)
        )
        if not self.detector_ready:
            self.face_cascade = None
            self.eye_cascades = []
        self.landmarker = None
        model_path = Path(__file__).resolve().parent / "models" / "face_landmarker.task"
        if mp is not None and mp_python is not None and mp_vision is not None and model_path.exists():
            options = mp_vision.FaceLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
                running_mode=mp_vision.RunningMode.VIDEO,
                num_faces=1,
                min_face_detection_confidence=0.55,
                min_face_presence_confidence=0.55,
                min_tracking_confidence=0.55,
            )
            self.landmarker = mp_vision.FaceLandmarker.create_from_options(options)
        self.closed_since: float | None = None
        self.was_closed = False
        self.blinks = 0
        self.yawn_since: float | None = None
        self.was_yawning = False
        self.yawns = 0
        self.started = time.monotonic()
        self.photo = None
        self.running = True
        self.paused = False
        self.after_id = None
        self.on_assessment = on_assessment

        self.image = tk.Label(self, bg="black")
        self.image.pack(fill="both", expand=True, padx=16, pady=(16, 8))
        self.status = tk.StringVar(value="Iniciando cámara...")
        tk.Label(
            self,
            textvariable=self.status,
            bg="#FFFFFF",
            fg="#1E2A3A",
            font=("Segoe UI", 12, "bold"),
            anchor="w",
            padx=12,
            pady=10,
        ).pack(fill="x", padx=16, pady=(0, 8))
        controls = tk.Frame(self, bg="#E9EEF5")
        controls.pack(fill="x", padx=16, pady=(0, 12))
        for text, command, color in (
            ("Pausar", self.pause, "#A9791E"),
            ("Reanudar", self.resume, "#1E7A46"),
            ("Detener cámara", self.stop_camera, "#A33A4C"),
            ("Cerrar", self.close, "#4A5D73"),
        ):
            tk.Button(
                controls, text=text, command=command, bg=color, fg="white",
                borderwidth=0, font=("Segoe UI", 12, "bold"),
                width=16 if text != "Cerrar" else 12,
                padx=18, pady=10,
            ).pack(side="left" if text != "Cerrar" else "right", padx=6)
        self.after_id = self.after(30, self.update_frame)

    @staticmethod
    def _open_camera():
        """Prueba cámaras y backends de Windows, validando un cuadro real."""
        backends = (
            ('MSMF', cv2.CAP_MSMF),
            ('DirectShow', cv2.CAP_DSHOW),
            ('Automático', cv2.CAP_ANY),
        )
        for backend_name, backend in backends:
            for index in (0, 1, 2):
                capture = cv2.VideoCapture(index, backend)
                if not capture.isOpened():
                    capture.release()
                    continue
                capture.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
                capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
                for _ in range(20):
                    ok, frame = capture.read()
                    if (ok and frame is not None and frame.size
                            and int(frame.max()) > 5):
                        return capture, frame, f'{backend_name}, índice {index}'
                    time.sleep(0.03)
                capture.release()
        return None, None, None

    @staticmethod
    def _point(landmarks, index, width, height):
        point = landmarks[index]
        return point.x * width, point.y * height

    @staticmethod
    def _distance(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def _eye_ratio(self, landmarks, indexes, width, height):
        points = [self._point(landmarks, index, width, height) for index in indexes]
        horizontal = self._distance(points[0], points[3])
        if horizontal == 0:
            return 0.0
        return (self._distance(points[1], points[5]) +
                self._distance(points[2], points[4])) / (2 * horizontal)

    def _mesh_metrics(self, frame):
        if self.landmarker is None:
            return None
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp = int(time.monotonic() * 1000)
        result = self.landmarker.detect_for_video(image, timestamp)
        if not result.face_landmarks:
            return None
        height, width = frame.shape[:2]
        landmarks = result.face_landmarks[0]
        left_eye = (33, 160, 158, 133, 153, 144)
        right_eye = (362, 385, 387, 263, 373, 380)
        ear = (self._eye_ratio(landmarks, left_eye, width, height) +
               self._eye_ratio(landmarks, right_eye, width, height)) / 2
        mouth_top = self._point(landmarks, 13, width, height)
        mouth_bottom = self._point(landmarks, 14, width, height)
        mouth_left = self._point(landmarks, 78, width, height)
        mouth_right = self._point(landmarks, 308, width, height)
        mouth_width = self._distance(mouth_left, mouth_right)
        mar = self._distance(mouth_top, mouth_bottom) / mouth_width if mouth_width else 0
        left_corner = self._point(landmarks, 33, width, height)
        right_corner = self._point(landmarks, 263, width, height)
        tilt = math.degrees(math.atan2(right_corner[1] - left_corner[1],
                                       right_corner[0] - left_corner[0]))
        xs = [point.x * width for point in landmarks]
        ys = [point.y * height for point in landmarks]
        box = (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))
        return ear, mar, tilt, box, landmarks

    def _update_temporal(self, eyes_closed, yawning, now):
        if eyes_closed and self.closed_since is None:
            self.closed_since = now
        elif not eyes_closed and self.closed_since is not None:
            duration = now - self.closed_since
            if self.was_closed and 0.08 <= duration <= 1.5:
                self.blinks += 1
            self.closed_since = None
        self.was_closed = eyes_closed
        if yawning and self.yawn_since is None:
            self.yawn_since = now
        elif not yawning and self.yawn_since is not None:
            if self.was_yawning and now - self.yawn_since >= 0.8:
                self.yawns += 1
            self.yawn_since = None
        self.was_yawning = yawning

    def update_frame(self):
        if not self.running:
            return
        if self.paused:
            return
        if self.first_frame is not None:
            frame = self.first_frame
            self.first_frame = None
            ok = True
        else:
            ok, frame = self.video.read()
        if not ok:
            self.status.set(
                f"No se pudo leer la cámara (índice {self.camera_index}). "
                "Revisa permisos o aplicaciones que la estén usando."
            )
            self.after_id = self.after(100, self.update_frame)
            return

        mesh_data = self._mesh_metrics(frame)
        if mesh_data is not None:
            ear, mar, tilt_degrees, box, landmarks = mesh_data
            now = time.monotonic()
            eyes_closed = ear < 0.21
            yawning = mar > 0.58
            self._update_temporal(eyes_closed, yawning, now)
            closed_seconds = 0.0 if self.closed_since is None else now - self.closed_since
            minutes = max((now - self.started) / 60.0, 1 / 60.0)
            signals = Signals(
                eye_closed_seconds=min(closed_seconds, 10.0),
                blinks_per_minute=round(self.blinks / minutes),
                yawns_per_minute=round(self.yawns / minutes),
                head_tilt_degrees=round(abs(tilt_degrees), 1),
                face_visible=True,
            )
            assessment = evaluate(signals)
            if self.on_assessment is not None:
                self.on_assessment(signals, assessment)
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (37, 99, 235), 2)
            for index in (33, 160, 158, 133, 153, 144,
                          362, 385, 387, 263, 373, 380):
                px, py = self._point(landmarks, index, frame.shape[1], frame.shape[0])
                cv2.circle(frame, (int(px), int(py)), 2, (13, 148, 136), -1)
            for index in (13, 14, 78, 308):
                px, py = self._point(landmarks, index, frame.shape[1], frame.shape[0])
                cv2.circle(frame, (int(px), int(py)), 3, (220, 80, 80), -1)
            cv2.putText(frame, assessment.level, (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (40, 80, 220), 2, cv2.LINE_AA)
            cv2.putText(frame, f"EAR {ear:.2f}  MAR {mar:.2f}  Inclinacion {abs(tilt_degrees):.1f} grados",
                        (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (40, 80, 220), 2, cv2.LINE_AA)
            ok, encoded = cv2.imencode(".png", cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if ok:
                png_data = base64.b64encode(encoded.tobytes()).decode("ascii")
                self.photo = tk.PhotoImage(data=png_data, format="png")
                self.image.configure(image=self.photo)
            self.status.set(
                f"MediaPipe | Nivel: {assessment.level} | "
                f"Ojos cerrados: {signals.eye_closed_seconds:.1f}s | "
                f"Parpadeos: {signals.blinks_per_minute}/min | "
                f"Bostezos: {signals.yawns_per_minute}/min"
            )
            self.after_id = self.after(30, self.update_frame)
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Mejora local para rostros oscuros por contraluz.
        gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        if self.detector_ready:
            faces = self.face_cascade.detectMultiScale(
                gray, 1.1, 5, minSize=(100, 100)
            )
        else:
            faces = []
        face_visible = True if not self.detector_ready else len(faces) > 0
        eyes_count = 0
        eye_centers = []
        tilt_degrees = 0.0

        if face_visible and self.detector_ready:
            x, y, w, h = max(faces, key=lambda item: item[2] * item[3])
            cv2.rectangle(frame, (x, y), (x + w, y + h), (37, 99, 235), 2)
            upper = gray[y:y + int(h * 0.60), x:x + w]
            min_eye = max(16, int(w * 0.08))
            detections = []
            for cascade in self.eye_cascades:
                found = cascade.detectMultiScale(
                    upper, 1.05, 3, minSize=(min_eye, min_eye)
                )
                if len(found) > len(detections):
                    detections = list(found)
            eyes = sorted(detections, key=lambda item: item[2] * item[3], reverse=True)[:2]
            eyes_count = len(eyes)
            for ex, ey, ew, eh in eyes:
                cv2.rectangle(frame, (x + ex, y + ey),
                              (x + ex + ew, y + ey + eh), (13, 148, 136), 2)
                eye_centers.append((x + ex + ew // 2, y + ey + eh // 2))
            if len(eye_centers) == 2:
                eye_centers.sort()
                left, right = eye_centers
                tilt_degrees = math.degrees(
                    math.atan2(right[1] - left[1], right[0] - left[0])
                )

        now = time.monotonic()
        # En una webcam de laptop un ojo puede perderse por reflejos;
        # menos de dos ojos detectados se trata como posible cierre.
        eyes_closed = self.detector_ready and face_visible and eyes_count < 2
        if eyes_closed and self.closed_since is None:
            self.closed_since = now
        elif not eyes_closed and self.closed_since is not None:
            if self.was_closed and now - self.closed_since >= 0.25:
                self.blinks += 1
            self.closed_since = None
        self.was_closed = eyes_closed

        closed_seconds = 0.0 if self.closed_since is None else now - self.closed_since
        minutes = max((now - self.started) / 60.0, 1 / 60.0)
        signals = Signals(
            eye_closed_seconds=min(closed_seconds, 10.0),
            blinks_per_minute=round(self.blinks / minutes),
            yawns_per_minute=0,
            head_tilt_degrees=round(tilt_degrees, 1),
            face_visible=face_visible,
        )
        assessment = evaluate(signals)
        if self.on_assessment is not None:
            self.on_assessment(signals, assessment)
        cv2.putText(frame, assessment.level, (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (40, 80, 220), 2, cv2.LINE_AA)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ok, encoded = cv2.imencode(".png", rgb)
        if ok:
            png_data = base64.b64encode(encoded.tobytes()).decode("ascii")
            self.photo = tk.PhotoImage(data=png_data, format="png")
            self.image.configure(image=self.photo)
        if self.detector_ready:
            self.status.set(
                "Haar (respaldo): falta face_landmarker.task. | "
                f"Nivel: {assessment.level} | Ojos cerrados: "
                f"{signals.eye_closed_seconds:.1f}s | "
                f"Parpadeos: {signals.blinks_per_minute}/min"
            )
        else:
            self.status.set(
                "Haar (respaldo) activo: boca y bostezos no disponibles. "
                "Descarga face_landmarker.task."
            )
        self.after_id = self.after(30, self.update_frame)

    def pause(self):
        if self.running:
            self.paused = True
            self.status.set("Cámara pausada. Pulsa Reanudar para continuar.")

    def resume(self):
        if not self.running:
            return
        if self.video is None:
            self.video, self.first_frame, self.camera_index = self._open_camera()
            if self.video is None:
                self.status.set("No se pudo reabrir la cámara.")
                return
        self.paused = False
        self.status.set("Cámara reanudada.")
        self.after_id = self.after(30, self.update_frame)

    def stop_camera(self):
        self.paused = True
        if self.video is not None:
            self.video.release()
            self.video = None
        self.status.set("Cámara detenida. Pulsa Reanudar para volver a abrirla.")

    def close(self):
        self.running = False
        if self.after_id is not None:
            try:
                self.after_cancel(self.after_id)
            except tk.TclError:
                pass
            self.after_id = None
        if self.landmarker is not None:
            self.landmarker.close()
        if getattr(self, "video", None) is not None:
            self.video.release()
        self.destroy()


def abrir_camara(master: tk.Misc, on_assessment=None):
    """Abre la captura sin cambiar el flujo de simulación existente."""
    return CameraWindow(master, on_assessment=on_assessment)
