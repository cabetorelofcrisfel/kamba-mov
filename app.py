from flask import Flask, request, jsonify, render_template
import os

app = Flask(__name__)

# Armazena mensagens em memória
mensagens_para_esp = []
mensagens_recebidas_esp = []

# =========================
# Página principal (HTML)
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
# Ver mensagens recebidas do ESP
# =========================
@app.route("/mensagens_esp", methods=["GET"])
def mensagens_esp():
    return jsonify({"mensagens": mensagens_recebidas_esp})

# =========================
# Execução gf
# =========================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))