import os
import cv2
import time
import json
import logging
import threading
import sqlite3
from datetime import datetime
from queue import Queue

# --- CONFIGURACIÓN DE ENTORNO PARA PADDLEOCR ---
os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'
os.environ['FLAGS_call_stack_level'] = '0'

from paddleocr import PaddleOCR
import paddle

# Forzar uso de CPU para estabilidad industrial
paddle.set_device('cpu')
logging.getLogger("ppocr").setLevel(logging.ERROR)

class VisionLabIndustrial:
    def __init__(self, filas=12, columnas=5, intervalo=15):
        self.FILAS = filas
        self.COLUMNAS = columnas
        self.INTERVALO = intervalo

        # --- GESTIÓN DE RUTAS DINÁMICAS (PORTABLE) ---
        # 1. Ruta de la carpeta donde está este main.py (src/)
        self.ruta_script = os.path.dirname(os.path.abspath(__file__))
        # 2. Ruta raíz del proyecto (TableScanner/)
        self.raiz = os.path.dirname(self.ruta_script)

        # 3. Definición de carpetas usando rutas relativas
        self.carpeta_base = os.path.join(self.raiz, "capturas")
        self.db_path = os.path.join(self.raiz, "data", "local_backup.db")
        self.ruta_modelos = os.path.join(self.ruta_script, "models")

        # Crear carpeta de capturas si no existe
        if not os.path.exists(self.carpeta_base):
            os.makedirs(self.carpeta_base)

        self.cola_procesamiento = Queue()
        self.activo = True

        print(f"\n>>> SISTEMA INICIADO <<<")
        print(f"[*] Carpeta de Trabajo: {self.raiz}")
        print(f"[*] Base de Datos: {self.db_path}")
        print(f"[*] Modelos IA: {self.ruta_modelos}")
        print("-" * 40)

        try:
            # Inicialización del OCR apuntando a la nueva carpeta interna 'src/models'
            self.ocr = PaddleOCR(
                det_model_dir=os.path.join(self.ruta_modelos, "det"),
                rec_model_dir=os.path.join(self.ruta_modelos, "rec"),
                cls_model_dir=os.path.join(self.ruta_modelos, "cls"),
                use_angle_cls=True,
                lang='en',
                show_log=False
            )
            print("[OK] Motores OCR cargados desde el interior del proyecto.")
        except Exception as e:
            print(f"[ERROR] No se encontraron los modelos en {self.ruta_modelos}: {e}")
            self.activo = False

    def hilo_captura(self):
        """Hilo encargado de capturar imágenes de la webcam"""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("[ERROR] No se puede acceder a la cámara.")
            self.activo = False
            return

        while self.activo:
            ret, frame = cap.read()
            if ret:
                ahora = datetime.now()
                nombre_foto = ahora.strftime("CAP_%Y%m%d_%H%M%S.jpg")
                ruta_foto = os.path.join(self.carpeta_base, nombre_foto)

                cv2.imwrite(ruta_foto, frame)
                self.cola_procesamiento.put(ruta_foto)
                print(f"[CAPTURA] Imagen guardada: {nombre_foto}")

            time.sleep(self.INTERVALO)

        cap.release()

    def hilo_ocr(self):
        """Hilo encargado de procesar imágenes y guardar en SQLite"""
        try:
            # Abrimos la conexión SQLite dentro del hilo para evitar errores de hilos
            conexion = sqlite3.connect(self.db_path, check_same_thread=False)
            cursor = conexion.cursor()
        except Exception as e:
            print(f"[ERROR] Fallo al conectar con la base de datos: {e}")
            return

        while self.activo:
            if not self.cola_procesamiento.empty():
                img_path = self.cola_procesamiento.get()

                try:
                    print(f"[OCR] Analizando: {os.path.basename(img_path)}...")
                    resultado = self.ocr.ocr(img_path, cls=True)

                    if resultado and resultado[0]:
                        # Limpiar y ordenar bloques de texto por posición (Y, X)
                        bloques = [line for line in resultado[0] if line]
                        bloques.sort(key=lambda x: (x[0][0][1], x[0][0][0]))

                        valores_finales = []
                        for res in bloques:
                            texto = res[1][0].replace(',', '.')
                            try:
                                valores_finales.append(float(texto))
                            except ValueError:
                                valores_finales.append(0.0)

                        # --- GUARDADO EN BD ---
                        # 1. Crear una nueva 'Toma'
                        tiempo_iso = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        cursor.execute(
                            "INSERT INTO tomas (tiempo_tomas) VALUES (?)",
                            (tiempo_iso,)
                        )
                        id_toma = cursor.lastrowid

                        # 2. Insertar los 60 valores vinculados a la toma
                        # id_variables va del 1 al 60
                        for i, valor in enumerate(valores_finales[:60]):
                            id_var = i + 1
                            cursor.execute(
                                """INSERT INTO tabla_datos
                                   (id_grupo_tomas, id_variables, promedio)
                                   VALUES (?, ?, ?)""",
                                (id_toma, id_var, valor)
                            )

                        conexion.commit()
                        print(f"[DB] ÉXITO: {len(valores_finales[:60])} valores guardados en Toma #{id_toma}")

                except Exception as e:
                    print(f"[ERROR] Error en el proceso OCR/DB: {e}")
                    conexion.rollback()

                self.cola_procesamiento.task_done()
            else:
                time.sleep(1)

        conexion.close()

    def ejecutar(self):
        """Lanza los hilos de trabajo"""
        t_cap = threading.Thread(target=self.hilo_captura, daemon=True)
        t_ocr = threading.Thread(target=self.hilo_ocr, daemon=True)

        t_cap.start()
        t_ocr.start()

if __name__ == "__main__":
    app = VisionLabIndustrial()
    try:
        if app.activo:
            app.ejecutar()
            print("\n" + "="*40)
            print(" SISTEMA DE LECTURA ACTIVO ")
            print(" (Presione Ctrl+C para salir) ")
            print("="*40 + "\n")

            while True:
                time.sleep(1)
    except KeyboardInterrupt:
        print("\n[!] Cerrando hilos y saliendo...")
        app.activo = False