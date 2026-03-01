from flask import Flask, request, jsonify, render_template, send_from_directory
import os
from datetime import datetime

app = Flask(__name__)

# =========================
# CONFIGURAÇÃO
# =========================
UPLOAD_FOLDER = "stream_images"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

mensagens_para_esp = []
mensagens_recebidas_esp = []

# =========================
# Página principal
# =========================
@app.route("/")
def home():
    return render_template("index.html")

# =========================
# API envia mensagem para ESP
# =========================
@app.route("/enviar", methods=["POST"])
def enviar():
    data = request.json
    mensagem = data.get("mensagem")

    if mensagem:
        mensagens_para_esp.append(mensagem)
        return jsonify({"status": "Mensagem enviada para ESP"})
    
    return jsonify({"erro": "Mensagem vazia"}), 400

# =========================
# ESP busca mensagens
# =========================
@app.route("/buscar", methods=["GET"])
def buscar():
    if mensagens_para_esp:
        mensagem = mensagens_para_esp.pop(0)
        return jsonify({"mensagem": mensagem})
    
    return jsonify({"mensagem": None})

# =========================
# ESP envia mensagem para API
# =========================
@app.route("/receber", methods=["POST"])
def receber():
    data = request.json
    mensagem = data.get("mensagem")

    if mensagem:
        mensagens_recebidas_esp.append(mensagem)
        return jsonify({"status": "Mensagem recebida com sucesso"})
    
    return jsonify({"erro": "Mensagem vazia"}), 400

# =========================
# Receber imagem do stream
# =========================
@app.route("/stream", methods=["POST"])
def receber_stream():

    if 'image' in request.files:
        # Caso envie como multipart/form-data
        image = request.files['image']
        filename = datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".jpg"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        image.save(filepath)

    else:
        # Caso envie binário puro (Content-Type: image/jpeg)
        image_data = request.data
        if not image_data:
            return jsonify({"erro": "Imagem vazia"}), 400

        filename = datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".jpg"
        filepath = os.path.join(UPLOAD_FOLDER, filename)

        with open(filepath, "wb") as f:
            f.write(image_data)

    return jsonify({"status": "Imagem recebida com sucesso"})

# =========================
# Listar imagens recebidas
# =========================
@app.route("/imagens", methods=["GET"])
def listar_imagens():
    arquivos = sorted(os.listdir(UPLOAD_FOLDER), reverse=True)
    return jsonify({"imagens": arquivos})

# =========================
# Servir imagem específica
# =========================
@app.route("/imagem/<filename>")
def mostrar_imagem(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

# =========================
# Ver mensagens recebidas do ESP
# =========================
@app.route("/mensagens_esp", methods=["GET"])
def mensagens_esp():
    return jsonify({"mensagens": mensagens_recebidas_esp})

# =========================
# Execução
# =========================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))