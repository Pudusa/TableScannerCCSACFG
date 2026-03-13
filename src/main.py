import os
import cv2
import time
import json
import logging
import threading
import sqlite3
from datetime import datetime
from queue import Queue

# --- CONFIGURACIÓN DE ENTORNO ---
os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'
os.environ['FLAGS_call_stack_level'] = '0'

from paddleocr import PaddleOCR
import paddle

paddle.set_device('cpu')
logging.getLogger("ppocr").setLevel(logging.ERROR)

class VisionLabIndustrial:
    def __init__(self, filas=12, columnas=5, intervalo=15):
        self.FILAS = filas
        self.COLUMNAS = columnas
        self.INTERVALO = intervalo
        self.carpeta_base = "capturas"
        self.cola_procesamiento = Queue()
        self.activo = True

        # Apunta directamente a tu BD ya creada
        self.db_path = "sistema_mediciones.db"

        ruta_modelos = r"C:\Bryan\Trabajo\Proyecto Camara\Prueba\modelos"

        print(">>> Cargando Motores OCR v2.7.3 (Offline)...")
        try:
            self.ocr = PaddleOCR(
                det_model_dir=os.path.join(ruta_modelos, "det"),
                rec_model_dir=os.path.join(ruta_modelos, "rec"),
                cls_model_dir=os.path.join(ruta_modelos, "cls"),
                use_angle_cls=True,
                use_gpu=False,
                lang='en',
                show_log=False
            )
            print(">>> Motores cargados con éxito desde disco duro.")
        except Exception as e:
            print(f"!!! Error al inicializar OCR: {e}")
            self.activo = False

        if not os.path.exists(self.carpeta_base):
            os.makedirs(self.carpeta_base)

    def limpiar_texto(self, txt):
        mapa = str.maketrans({'O':'0','o':'0','D':'0','I':'1','l':'1','|':'1','S':'5','s':'5','B':'8','Z':'2',',':'.'})
        t = txt.translate(mapa)
        solo_num = "".join([c for c in t if c in "0123456789.-"])
        try:
            if not solo_num or solo_num == "-": return 0
            return float(solo_num) if "." in solo_num else int(solo_num)
        except: return solo_num

    def hilo_captura(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("!!! Error: No se pudo abrir la cámara.")
            self.activo = False
            return

        print(f">>> Hilo de cámara activo (Frecuencia: {self.INTERVALO}s)")
        while self.activo:
            t_inicio = time.time()
            ret, frame = cap.read()
            if ret:
                ahora = datetime.now()
                ruta_final = os.path.join(self.carpeta_base, ahora.strftime("%Y-%m-%d"), ahora.strftime("%H-%M"))
                os.makedirs(ruta_final, exist_ok=True)

                nombre_img = f"toma_{ahora.strftime('%H-%M-%S')}.jpg"
                path_img = os.path.join(ruta_final, nombre_img)
                cv2.imwrite(path_img, frame)

                self.cola_procesamiento.put(path_img)

            time.sleep(max(0, self.INTERVALO - (time.time() - t_inicio)))
        cap.release()

    def hilo_ocr(self):
        print(">>> Analizador OCR conectado a BD esperando imágenes...")

        # Conexión dedicada para este hilo
        conexion = sqlite3.connect(self.db_path)
        cursor = conexion.cursor()

        while self.activo:
            if not self.cola_procesamiento.empty():
                img_path = self.cola_procesamiento.get()
                try:
                    resultado = self.ocr.ocr(img_path, cls=True)
                    if resultado and resultado[0]:

                        # Extraer valores manteniendo el orden natural del OCR
                        valores_planos = []
                        for linea in resultado[0]:
                            coords, (texto, conf) = linea
                            valor_limpio = self.limpiar_texto(texto)
                            valores_planos.append(valor_limpio)

                        # Reconstruir la matriz para el JSON de la tabla 'tomas'
                        matriz = []
                        for i in range(0, len(valores_planos), self.COLUMNAS):
                            fila = valores_planos[i : i + self.COLUMNAS]
                            matriz.append(fila)

                        # --- INSERCIÓN EN LA BASE DE DATOS ---
                        fecha_actual = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                        # 1. Guardar el bloque general en 'tomas'
                        cursor.execute(
                            "INSERT INTO tomas (grupo_tomas, tiempo_tomas) VALUES (?, ?)",
                            (json.dumps(matriz), json.dumps([fecha_actual]))
                        )
                        id_toma = cursor.lastrowid

                        # 2. Iterar e insertar secuencialmente en 'tabla_datos'
                        for indice, valor in enumerate(valores_planos):
                            id_variable_actual = indice + 1 # Asigna IDs del 1 en adelante secuencialmente
                            cursor.execute(
                                "INSERT INTO tabla_datos (id_grupo_tomas, id_variables, promedio) VALUES (?, ?, ?)",
                                (id_toma, id_variable_actual, valor)
                            )

                        # Confirmar la transacción
                        conexion.commit()

                        print(f"[OK] {os.path.basename(img_path)}: Procesada y guardada en BD ({len(valores_planos)} lecturas)")
                except Exception as e:
                    print(f"!!! Error procesando {img_path}: {e}")
                    conexion.rollback() # Revierte los cambios si falla a la mitad para evitar datos corruptos
                self.cola_procesamiento.task_done()
            else:
                time.sleep(1)

        conexion.close()

    def ejecutar(self):
        threading.Thread(target=self.hilo_captura, daemon=True).start()
        threading.Thread(target=self.hilo_ocr, daemon=True).start()

if __name__ == "__main__":
    app = VisionLabIndustrial()
    try:
        if app.activo:
            app.ejecutar()
            print("--- SISTEMA EN MARCHA (Ctrl+C para salir) ---")
            while True: time.sleep(1)
    except KeyboardInterrupt:
        print("\n>>> Finalizando sistema...")
        app.activo = False