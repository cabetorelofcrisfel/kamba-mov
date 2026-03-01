import customtkinter as ctk
import cv2
import torch
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
from numpy.linalg import norm
from PIL import Image, ImageTk
import threading
import time

# ==============================
# CONFIGURAÇÕES
# ==============================
DROIDCAM_URL = "http://192.168.42.129:4747/video"
THRESHOLD_SIMILARITY = 0.75
COOLDOWN = 2
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ==============================
# MODELOS
# ==============================
mtcnn = MTCNN(image_size=160, margin=20, device=DEVICE)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(DEVICE)

# ==============================
# APP
# ==============================
class FaceApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Sistema de Reconhecimento Facial - Alta Precisão")
        self.geometry("1000x700")
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.video_label = ctk.CTkLabel(self, text="")
        self.video_label.pack(pady=20)

        self.counter_label = ctk.CTkLabel(
            self,
            text="Rostos diferentes: 0",
            font=("Arial", 24, "bold")
        )
        self.counter_label.pack(pady=10)

        self.start_button = ctk.CTkButton(
            self,
            text="Iniciar",
            command=self.start_camera,
            width=200,
            height=40
        )
        self.start_button.pack(pady=10)

        self.stop_button = ctk.CTkButton(
            self,
            text="Parar",
            command=self.stop_camera,
            width=200,
            height=40,
            fg_color="red"
        )
        self.stop_button.pack(pady=10)

        self.cap = None
        self.running = False
        self.known_faces = []
        self.last_detection_time = 0

    # ==========================
    # FUNÇÃO DE COMPARAÇÃO
    # ==========================
    def is_new_face(self, embedding):
        if len(self.known_faces) == 0:
            return True
        
        for known in self.known_faces:
            similarity = np.dot(embedding, known) / (norm(embedding) * norm(known))
            if similarity > THRESHOLD_SIMILARITY:
                return False
        return True

    # ==========================
    # INICIAR CÂMERA
    # ==========================
    def start_camera(self):
        self.cap = cv2.VideoCapture(DROIDCAM_URL)
        if not self.cap.isOpened():
            print("Erro ao conectar ao DroidCam.")
            return
        
        self.running = True
        threading.Thread(target=self.update_frame, daemon=True).start()

    # ==========================
    # PARAR CÂMERA
    # ==========================
    def stop_camera(self):
        self.running = False
        if self.cap:
            self.cap.release()

    # ==========================
    # LOOP DE VÍDEO
    # ==========================
    def update_frame(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            boxes, _ = mtcnn.detect(rgb_frame)

            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box)

                    face = rgb_frame[y1:y2, x1:x2]

                    try:
                        face_tensor = mtcnn(face)
                        if face_tensor is None:
                            continue

                        face_tensor = face_tensor.unsqueeze(0).to(DEVICE)

                        with torch.no_grad():
                            embedding = resnet(face_tensor)

                        embedding = embedding.cpu().numpy()[0]

                        current_time = time.time()

                        if current_time - self.last_detection_time > COOLDOWN:
                            if self.is_new_face(embedding):
                                self.known_faces.append(embedding)
                                self.last_detection_time = current_time

                                self.counter_label.configure(
                                    text=f"Rostos diferentes: {len(self.known_faces)}"
                                )

                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

                    except:
                        continue

            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

            time.sleep(0.01)


if __name__ == "__main__":
    app = FaceApp()
    app.mainloop()