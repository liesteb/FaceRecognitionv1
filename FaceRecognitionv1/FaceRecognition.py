# ====================ErickMoya======================Comienzo
import cv2
import face_recognition as fr
import os
import numpy
from datetime import datetime
import winsound
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import threading 

# ==========================================
# CONFIGURACION
# ==========================================
ruta_empleados = 'Empleados'
archivo_registro = 'registro.csv'

# Creamos las carpetas si no existen
if not os.path.exists(ruta_empleados):
    os.makedirs(ruta_empleados)

if not os.path.exists(archivo_registro):
    with open(archivo_registro, 'w') as f:
        f.write('Nombre,Hora,Fecha,Tipo\n')

# Variable para controlar el tiempo entre registros (evitar duplicados inmediatos)
ultimo_registro = {}

# ==========================================
# FUNCIONES AUXILIARES
# ==========================================

def sanitizar_imagen(imagen_bgr):
    """Limpia la imagen para evitar errores de compatibilidad con dlib"""
    try:
        imagen_rgb = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2RGB)
        imagen_rgb = imagen_rgb.astype('uint8')
        imagen_rgb = imagen_rgb[:, :, :3]
        imagen_rgb = numpy.ascontiguousarray(imagen_rgb)
        return imagen_rgb
    except Exception:
        return None

def cargar_codificaciones():
    """Carga todas las fotos de la carpeta al inicio"""
    imagenes = []
    nombres = []
    lista_archivos = os.listdir(ruta_empleados)

    print("Cargando rostros...")
    for archivo in lista_archivos:
        if not archivo.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        
        ruta_completa = os.path.join(ruta_empleados, archivo)
        imagen_cv = cv2.imread(ruta_completa)
        
        if imagen_cv is None:
            continue

        imagen_rgb = sanitizar_imagen(imagen_cv)
        if imagen_rgb is not None:
            codigos = fr.face_encodings(imagen_rgb)
            if codigos:
                imagenes.append(codigos[0])
                nombres.append(os.path.splitext(archivo)[0])
                print(f"-> Cargado: {os.path.splitext(archivo)[0]}")
    
    return imagenes, nombres

# Cargamos la base de datos antes de iniciar la ventana
bases_codificadas, nombres_empleados = cargar_codificaciones()

def reproducir_sonido():
    """Ejecuta el sonido en un hilo separado para no congelar el video"""
    winsound.Beep(1000, 200)

def registrar_evento(nombre):
    """Logica para registrar entrada o salida en el CSV"""
    ahora = datetime.now()
    fecha_hoy = ahora.strftime('%Y-%m-%d')
    hora_actual = ahora.strftime('%H:%M:%S')
    
    # Evitar registros multiples en menos de 30 segundos
    if nombre in ultimo_registro:
        tiempo_pasado = (ahora - ultimo_registro[nombre]).total_seconds()
        if tiempo_pasado < 30:
            return None 
    
    # Determinar si es Entrada o Salida
    tipo_evento = "Entrada"
    try:
        if os.path.exists(archivo_registro):
            with open(archivo_registro, 'r') as f:
                lineas = f.readlines()
            
            eventos_hoy = []
            for linea in lineas:
                datos = linea.strip().split(',')
                if len(datos) >= 4:
                    if datos[0] == nombre and datos[2] == fecha_hoy:
                        eventos_hoy.append(datos[3])
            
            if eventos_hoy and eventos_hoy[-1] == "Entrada":
                tipo_evento = "Salida"
    except Exception:
        pass

    # Guardar en CSV
    with open(archivo_registro, 'a') as f:
        f.write(f'{nombre},{hora_actual},{fecha_hoy},{tipo_evento}\n')
    
    ultimo_registro[nombre] = ahora
    
    # Lanzar sonido en segundo plano
    hilo_sonido = threading.Thread(target=reproducir_sonido)
    hilo_sonido.start()
    
    return f"{nombre} - {tipo_evento} - {hora_actual}"

# ====================ErickMoya======================END

# ==========================================
# INTERFAZ GRAFICA (OPTIMIZADA)
# ==========================================

class AplicacionAsistencia:
    def __init__(self, ventana, titulo):
        self.ventana = ventana
        self.ventana.title(titulo)
        self.ventana.geometry("800x650")
        
        # Elementos de la interfaz
        tk.Label(self.ventana, text="Sistema de Control de Asistencia", font=("Arial", 16, "bold")).pack(pady=10)
        
        self.video_label = tk.Label(self.ventana)
        self.video_label.pack()
        
        self.info_label = tk.Label(self.ventana, text="Esperando detección...", font=("Arial", 12), fg="blue")
        self.info_label.pack(pady=10)

        self.lista_registros = tk.Listbox(self.ventana, width=80, height=8)
        self.lista_registros.pack(pady=5)

        # Inicializar camara
        self.captura = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        
        # OPTIMIZACION: Variables para 'saltar' frames
        self.procesar_este_frame = True # Interruptor para procesar uno si, uno no
        self.ubicaciones_guardadas = []
        self.nombres_guardados = []
        self.colores_guardados = []
        
        # Iniciar bucle
        self.actualizar_frame()

    def actualizar_frame(self):
        exito, cuadro = self.captura.read()
        
        if exito:
            # OPTIMIZACION 1: Reducir tamaño para el reconocimiento (no afecta visualización)
            cuadro_peq = cv2.resize(cuadro, (0, 0), None, 0.25, 0.25)
            
            # Solo procesamos reconocimiento si el interruptor es True
            if self.procesar_este_frame:
                cuadro_rgb_proc = sanitizar_imagen(cuadro_peq)
                
                # Reiniciamos listas temporales
                self.ubicaciones_guardadas = []
                self.nombres_guardados = []
                self.colores_guardados = []

                if cuadro_rgb_proc is not None:
                    caras_loc = fr.face_locations(cuadro_rgb_proc)
                    # Solo codificamos si hay caras (ahorra tiempo)
                    if caras_loc:
                        caras_cod = fr.face_encodings(cuadro_rgb_proc, caras_loc)

                        for codif, ubic in zip(caras_cod, caras_loc):
                            matches = fr.compare_faces(bases_codificadas, codif)
                            distancias = fr.face_distance(bases_codificadas, codif)
                            
                            mejor_indice = numpy.argmin(distancias)
                            nombre = "Desconocido"
                            color = (0, 0, 255) # Rojo

                            if matches[mejor_indice] and distancias[mejor_indice] < 0.6:
                                nombre = nombres_empleados[mejor_indice]
                                color = (0, 255, 0) # Verde
                                
                                # Registrar
                                mensaje = registrar_evento(nombre)
                                if mensaje:
                                    self.lista_registros.insert(0, mensaje)
                                    self.info_label.config(text=f"Registrado: {mensaje}", fg="green")

                            # Guardamos los datos para usarlos en el siguiente frame (que no se procesara)
                            self.ubicaciones_guardadas.append(ubic)
                            self.nombres_guardados.append(nombre)
                            self.colores_guardados.append(color)

            # Cambiamos el interruptor para el siguiente ciclo
            # Si era True ahora es False, y viceversa
            self.procesar_este_frame = not self.procesar_este_frame

            # DIBUJAR RESULTADOS (Usamos los datos guardados, sean nuevos o del frame anterior)
            for (top, right, bottom, left), nombre, color in zip(self.ubicaciones_guardadas, self.nombres_guardados, self.colores_guardados):
                # Escalamos x4 porque procesamos al 0.25
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                cv2.rectangle(cuadro, (left, top), (right, bottom), color, 2)
                cv2.rectangle(cuadro, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
                cv2.putText(cuadro, nombre, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)

            # Convertir para Tkinter
            img_color = cv2.cvtColor(cuadro, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_color)
            img_tk = ImageTk.PhotoImage(image=img_pil)
            
            self.video_label.imgtk = img_tk
            self.video_label.configure(image=img_tk)
        
        # Llamar de nuevo en 10ms 
        self.ventana.after(10, self.actualizar_frame)

    def cerrar(self):
        self.captura.release()
        self.ventana.destroy()

if __name__ == "__main__":
    if not bases_codificadas:
        print("ADVERTENCIA: No hay rostros cargados.")
    
    root = tk.Tk()
    app = AplicacionAsistencia(root, "Reconocimiento Facial ")
    root.protocol("WM_DELETE_WINDOW", app.cerrar)
    root.mainloop()