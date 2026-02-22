# contador_pessoas_completo.py
import customtkinter as ctk
from tkinter import messagebox
from PIL import Image, ImageTk
import cv2
from ultralytics import YOLO
import threading
import subprocess
import time
from flask import Flask, jsonify
import logging

# -------- CONFIGURAÇÕES FLASK -------- #
app = Flask(__name__)
total_pessoas = 0

# Desliga logs do Flask para não poluir o terminal
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

# -------- CONFIGURAÇÕES YOLO -------- #
model = YOLO("yolov8m.pt")  # modelo YOLOv8 médio
cap = None  # câmera será inicializada depois

# Função de contagem contínua
def contador_pessoas():
    global total_pessoas, cap
    while True:
        if cap is None or not cap.isOpened():
            time.sleep(0.5)
            continue

        ret, frame = cap.read()
        if not ret:
            continue

        # Reduz resolução para detecção mais rápida
        frame_small = cv2.resize(frame, (640, 360))
        results = model.predict(source=frame_small, conf=0.5, classes=[0])

        count = 0
        for r in results:
            for box in r.boxes:
                count += 1

        total_pessoas = count
        time.sleep(0.1)  # 10 FPS

# Roda o contador em background
threading.Thread(target=contador_pessoas, daemon=True).start()

# Endpoint Flask
@app.route("/pessoas")
def get_pessoas():
    global total_pessoas
    return jsonify({"total": total_pessoas})

# Função para rodar Flask em background
def rodar_flask():
    app.run(host="0.0.0.0", port=5000)

threading.Thread(target=rodar_flask, daemon=True).start()

# -------- INTERFACE CUSTOMTKINTER -------- #
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")

class ContadorPessoasApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Contador Futurista de Pessoas - YOLOv8")
        self.geometry("900x600")

        self.cap = None
        self.running = False

        # Frames
        self.frame_video = ctk.CTkFrame(self, corner_radius=10)
        self.frame_video.pack(pady=20, padx=20, fill="both", expand=True)

        self.frame_controls = ctk.CTkFrame(self, corner_radius=10)
        self.frame_controls.pack(pady=10, padx=20, fill="x")

        # Canvas vídeo
        self.label_video = ctk.CTkLabel(self.frame_video, text="")
        self.label_video.pack(expand=True)

        # Inputs IP/Porta
        self.ip_entry = ctk.CTkEntry(self.frame_controls, placeholder_text="IP do DroidCam")
        self.ip_entry.pack(side="left", padx=10)
        self.port_entry = ctk.CTkEntry(self.frame_controls, placeholder_text="Porta")
        self.port_entry.pack(side="left", padx=10)

        # Botões
        self.btn_connect = ctk.CTkButton(self.frame_controls, text="Conectar DroidCam", command=self.connect_droidcam)
        self.btn_connect.pack(side="left", padx=10)

        self.btn_start = ctk.CTkButton(self.frame_controls, text="Iniciar Contagem", command=self.start)
        self.btn_start.pack(side="left", padx=10)

        self.btn_stop = ctk.CTkButton(self.frame_controls, text="Parar Contagem", command=self.stop)
        self.btn_stop.pack(side="left", padx=10)

        self.total_label = ctk.CTkLabel(self.frame_controls, text="Total de pessoas: 0")
        self.total_label.pack(side="right", padx=10)

        self.try_open_camera()

    def try_open_camera(self):
        global cap
        cap = cv2.VideoCapture("/dev/video0")
        self.cap = cap
        if not cap.isOpened():
            messagebox.showinfo("Info", "Câmera não detectada. Conecte DroidCam via IP/Porta.")

    def connect_droidcam(self):
        ip = self.ip_entry.get()
        port = self.port_entry.get()
        if not ip or not port:
            messagebox.showwarning("Erro", "IP e Porta são obrigatórios!")
            return

        try:
            subprocess.Popen(
                ["./droidcam-cli", ip, port, "-v"], 
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            time.sleep(2)
            self.try_open_camera()
            if self.cap.isOpened():
                messagebox.showinfo("Sucesso", "DroidCam conectada!")
            else:
                messagebox.showerror("Erro", "Não foi possível abrir a câmera /dev/video0")
        except Exception as e:
            messagebox.showerror("Erro", f"Falha ao conectar DroidCam: {e}")

    def start(self):
        if not self.cap or not self.cap.isOpened():
            messagebox.showerror("Erro", "Nenhuma câmera disponível!")
            return
        self.running = True
        threading.Thread(target=self.update_frame, daemon=True).start()

    def stop(self):
        self.running = False

    def update_frame(self):
        global total_pessoas
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue

            # Reduz resolução para detecção
            frame_small = cv2.resize(frame, (640, 360))
            results = model.predict(source=frame_small, conf=0.5, classes=[0])

            count = 0
            for r in results:
                for box in r.boxes:
                    count += 1
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    h_ratio = frame.shape[0] / 360
                    w_ratio = frame.shape[1] / 640
                    x1, y1, x2, y2 = int(x1*w_ratio), int(y1*h_ratio), int(x2*w_ratio), int(y2*h_ratio)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                    cv2.putText(frame, "Pessoa", (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

            pessoas_texto = f"{count} pessoa" if count == 1 else f"{count} pessoas"
            self.total_label.configure(text=f"Total de pessoas: {pessoas_texto}")

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            self.label_video.imgtk = imgtk
            self.label_video.configure(image=imgtk)

# -------- RODA A INTERFACE -------- #
if __name__ == "__main__":
    app = ContadorPessoasApp()
    app.mainloop()
