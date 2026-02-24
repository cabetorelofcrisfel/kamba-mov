from flask import Flask, request, render_template, send_from_directory, jsonify
from flask_socketio import SocketIO
import os
from datetime import datetime

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 20 * 1024 * 1024  # limite 20MB

# Configuração compatível com Railway/Render
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# Criar pasta uploads se não existir
if not os.path.exists("uploads"):
    os.makedirs("uploads")

dados = []
mensagens_esp = []

# ================= DASHBOARD =================
@app.route("/")
def index():
    return jsonify({
        "status": "API online",
        "registros": len(dados)
    })

# ================= PING =================
@app.route("/ping")
def ping():
    return "pong", 200

# ================= RECEBER DO ESP =================
@app.route("/upload", methods=["POST"])
def upload():
    texto = request.form.get("texto")
    arquivo = request.files.get("arquivo")

    print("=== Novo envio recebido ===")
    print("Texto:", texto)
    print("Arquivo:", arquivo.filename if arquivo else None)

    nome_arquivo = None

    if arquivo:
        nome_arquivo = datetime.now().strftime("%Y%m%d%H%M%S_") + arquivo.filename
        caminho = os.path.join(app.config["UPLOAD_FOLDER"], nome_arquivo)
        arquivo.save(caminho)
        print("Arquivo salvo em:", caminho)

    registro = {
        "texto": texto,
        "arquivo": nome_arquivo,
        "data": datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    }

    dados.insert(0, registro)

    # Notifica dashboard em tempo real
    socketio.emit("novo", registro)

    return jsonify({"status": "recebido"})

# ================= ENVIAR MENSAGEM PARA ESP =================
@app.route("/send_message", methods=["POST"])
def send_message():
    msg = request.form.get("texto")

    if msg:
        mensagens_esp.append(msg)
        print("Mensagem enviada para ESP:", msg)
        return jsonify({"status": "ok"})
    else:
        return jsonify({"status": "erro", "motivo": "texto vazio"}), 400

# ================= ESP BUSCA MENSAGENS =================
@app.route("/get_messages")
def get_messages():
    global mensagens_esp
    msgs = mensagens_esp.copy()
    mensagens_esp = []  # limpa após envio

    if msgs:
        print("ESP buscou mensagens:", msgs)

    return jsonify(msgs)

# ================= SERVIR ARQUIVOS =================
@app.route("/uploads/<path:nome>")
def arquivo(nome):
    return send_from_directory("uploads", nome)

# ================= INICIAR SERVIDOR =================
if __name__ == "__main__":
    print("=== API Iniciada ===")

    port = int(os.environ.get("PORT", 5000))
    socketio.run(app, host="0.0.0.0", port=port)