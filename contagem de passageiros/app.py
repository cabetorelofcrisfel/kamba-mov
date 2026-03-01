import cv2
import threading
import queue
import time
import numpy as np
import customtkinter as ctk
from PIL import Image, ImageTk
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from numpy.linalg import norm
import json
import os
from datetime import datetime
import math
import gc

# ═══════════════════════════════════════════════════════════
#  CONFIGURAÇÃO
# ═══════════════════════════════════════════════════════════
VELOCIDADE_MINIMA    = 3.0
TEMPO_MIN_PRESENCA   = 30.0
TEMPO_AUSENCIA_SAIDA = 60.0
VALOR_DESCIDA        = 300

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"[INFO] Dispositivo: {DEVICE}")

# ── limiares de reconhecimento ─────────────────────────────
#
#  FaceNet VGGFace2 — similaridade coseno entre pessoas:
#    Mesma pessoa, boa qualidade  : 0.85 – 1.00
#    Mesma pessoa, pose diferente : 0.75 – 0.90
#    Pessoas diferentes           : 0.00 – 0.65  ← separação clara
#
#  LIMIAR_MATCH = 0.75
#    Só aceita como "mesma pessoa" se sim >= 0.75.
#    Conservador de propósito — preferimos criar um ID a mais
#    do que misturar dois rostos diferentes.
#
#  LIMIAR_REJEICAO = 0.60
#    Abaixo disto é definitivamente pessoa diferente.
#    Zona de dúvida: 0.60 – 0.75 (usa votos + rastreador).
#
#  MARGEM_SEPARACAO = 0.12
#    Para aceitar um match, o melhor UID tem de estar
#    pelo menos 0.12 ACIMA do segundo melhor.
#    Elimina ambiguidade quando duas pessoas são parecidas.
#
LIMIAR_MATCH      = 0.75   # acima → mesma pessoa (match seguro)
LIMIAR_REJEICAO   = 0.60   # abaixo → pessoa diferente
MARGEM_SEPARACAO  = 0.10   # diferença mínima entre 1º e 2º lugar

VOTOS_MINIMOS   = 3      # votos para confirmar zona de dúvida
COOLDOWN_NOVO   = 4.0    # segundos entre criar IDs novos
FRAMES_CONF     = 3      # frames para confirmar entrada

MIN_FACE_PX     = 60     # tamanho mínimo do rosto em pixels

# ╔══════════════════════════════════════════════════════════╗
# ║  APRENDIZAGEM PROGRESSIVA                               ║
# ║                                                         ║
# ║  O sistema divide o tempo de visão em 3 fases:         ║
# ║                                                         ║
# ║  FASE 1 — ENTRADA (0–5s)                               ║
# ║    Recolhe 1 embedding a cada 1.5s                     ║
# ║    Objectivo: identificar rapidamente                   ║
# ║    MAX_EMB_FASE1 = 4 embeddings                        ║
# ║                                                         ║
# ║  FASE 2 — CONSOLIDAÇÃO (5–20s)                         ║
# ║    Recolhe 1 embedding a cada 0.8s                     ║
# ║    Objectivo: reforçar com mais ângulos/expressões      ║
# ║    MAX_EMB_FASE2 = 12 embeddings acumulados            ║
# ║                                                         ║
# ║  FASE 3 — FORTALECIMENTO (20s+)                        ║
# ║    Recolhe 1 embedding a cada 0.4s                     ║
# ║    Objectivo: máxima cobertura de poses/luz             ║
# ║    MAX_EMB_TOTAL = 30 embeddings                       ║
# ║                                                         ║
# ║  Cada novo embedding só é aceite se for                 ║
# ║  DIVERSO dos já existentes (sim < 0.92)                ║
# ║  → evita guardar duplicados do mesmo ângulo             ║
# ╚══════════════════════════════════════════════════════════╝

# Fases de aprendizagem
FASES = [
    # (duração_seg, intervalo_seg, nome)
    (5.0,  1.5,  "ENTRADA"),        # fase 1: rápida, poucos embeddings
    (15.0, 0.8,  "CONSOLIDAÇÃO"),   # fase 2: reforço moderado
    (9999, 0.4,  "FORTALECIMENTO"), # fase 3: cobertura máxima (infinita)
]
MAX_EMB_TOTAL   = 30    # máximo de embeddings por pessoa
SIM_MIN_DIVERSO = 0.92  # novo embedding só entra se sim < este valor
                        # (garante diversidade — não duplica o mesmo ângulo)

FUNDO_MODO    = "off"
FUNDO_ESCURO  = 0.15
EXPAND_X      = 2.2
EXPAND_Y_UP   = 0.8
EXPAND_Y_DOWN = 5.0
MARGEM_BLEND  = 18


# ═══════════════════════════════════════════════════════════
#  MATEMÁTICA
# ═══════════════════════════════════════════════════════════
def _sim(a, b):
    na, nb = norm(a), norm(b)
    if na < 1e-9 or nb < 1e-9: return 0.0
    return float(np.dot(a, b) / (na * nb))

def _sim_max(emb, lista):
    if not lista: return 0.0
    return max(_sim(emb, it["emb"]) for it in lista)

def _sim_media(emb, lista):
    if not lista: return 0.0
    vals  = [_sim(emb, it["emb"]) * it["q"] for it in lista]
    pesos = [it["q"] for it in lista]
    return sum(vals) / sum(pesos) if sum(pesos) > 0 else 0.0

def _votos(emb, lista, limiar):
    return sum(1 for it in lista if _sim(emb, it["emb"]) >= limiar)

def _e_diverso(emb, lista, limiar_diversidade=SIM_MIN_DIVERSO):
    """
    Retorna True se o embedding é suficientemente diferente
    de todos os já guardados. Evita guardar duplicados do mesmo ângulo.
    """
    if not lista: return True
    return _sim_max(emb, lista) < limiar_diversidade


# ═══════════════════════════════════════════════════════════
#  SEGMENTAÇÃO DE FUNDO
# ═══════════════════════════════════════════════════════════
def aplicar_fundo(frame, bboxes, modo):
    if modo == "off" or not bboxes: return frame
    H, W = frame.shape[:2]
    m = np.zeros((H, W), dtype=np.float32)
    for (x, y, w, h) in bboxes:
        cx = x + w // 2
        mw = int(w * EXPAND_X / 2)
        x1=max(0,cx-mw); x2=min(W,cx+mw)
        y1=max(0,int(y-h*EXPAND_Y_UP)); y2=min(H,int(y+h*EXPAND_Y_DOWN))
        if x2>x1 and y2>y1: m[y1:y2,x1:x2]=1.0
    if m.max()==0: return frame
    k = MARGEM_BLEND*2+1
    m = cv2.GaussianBlur(m,(k,k),MARGEM_BLEND/2.0)
    mx=m.max()
    if mx>0: m=np.clip(m/mx,0,1)
    if   modo=="blur": fundo=cv2.GaussianBlur(frame,(15,15),0)
    elif modo=="dark": fundo=(frame*FUNDO_ESCURO).astype(np.uint8)
    elif modo=="bw":   fundo=cv2.cvtColor(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY),cv2.COLOR_GRAY2BGR)
    else: return frame
    m3=m[:,:,np.newaxis]
    return (frame.astype(np.float32)*m3+fundo.astype(np.float32)*(1-m3)).astype(np.uint8)


# ═══════════════════════════════════════════════════════════
#  GPS
# ═══════════════════════════════════════════════════════════
def haversine(lat1,lon1,lat2,lon2):
    R=6371.0; p1,p2=math.radians(lat1),math.radians(lat2)
    dp,dl=math.radians(lat2-lat1),math.radians(lon2-lon1)
    a=math.sin(dp/2)**2+math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return R*2*math.atan2(math.sqrt(a),math.sqrt(1-a))


# ═══════════════════════════════════════════════════════════
#  MOTOR FACIAL — MTCNN + FaceNet
# ═══════════════════════════════════════════════════════════
class MotorFacial:
    def __init__(self):
        self._fila    = queue.Queue(maxsize=2)
        self._cache   = []
        self._lock    = threading.Lock()
        self._fps_det = 0.0
        self._pronto  = False
        threading.Thread(target=self._carregar, daemon=True).start()

    def _carregar(self):
        print(f"[MOTOR] Carregando MTCNN + FaceNet ({DEVICE})...")
        try:
            self.mtcnn_det = MTCNN(
                keep_all=True, min_face_size=MIN_FACE_PX,
                thresholds=[0.6,0.7,0.7], device=DEVICE, post_process=False)
            self.mtcnn_aln = MTCNN(
                image_size=160, margin=20, keep_all=False,
                thresholds=[0.6,0.7,0.7], device=DEVICE, post_process=True)
            self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(DEVICE)
            self._pronto = True
            print("[MOTOR] ✅ Pronto!")
            threading.Thread(target=self._worker, daemon=True).start()
        except Exception as e:
            print(f"[MOTOR] ❌ {e}")

    def enviar_frame(self, frame):
        try: self._fila.get_nowait()
        except queue.Empty: pass
        try: self._fila.put_nowait(frame.copy())
        except queue.Full: pass

    def _worker(self):
        fc=0
        while True:
            try: frame=self._fila.get(timeout=1.0)
            except queue.Empty: continue
            fc+=1
            if fc%30==0: gc.collect()
            t0=time.time()
            try:
                res=self._processar(frame)
                with self._lock: self._cache=res
                dt=time.time()-t0
                self._fps_det=round(1.0/dt,1) if dt>0 else 0
            except Exception as e:
                print(f"[MOTOR] ❌ {e}")
                import traceback; traceback.print_exc()
                with self._lock: self._cache=[]

    def _processar(self, frame_bgr):
        img=Image.fromarray(cv2.cvtColor(frame_bgr,cv2.COLOR_BGR2RGB))
        boxes,probs=self.mtcnn_det.detect(img)
        if boxes is None: return []
        saida=[]
        for box,prob in zip(boxes,probs):
            if prob is None or prob<0.85: continue
            x1,y1,x2,y2=int(box[0]),int(box[1]),int(box[2]),int(box[3])
            x1=max(0,x1); y1=max(0,y1)
            x2=min(img.width,x2); y2=min(img.height,y2)
            w=x2-x1; h=y2-y1
            if w<MIN_FACE_PX or h<MIN_FACE_PX: continue
            emb=self._embedding(img,x1,y1,x2,y2)
            if emb is None: continue
            saida.append({"bbox":(x1,y1,w,h),"emb":emb,"conf":float(prob),"q":float(prob)})
        return saida

    def _embedding(self, img, x1, y1, x2, y2):
        try:
            mg=20
            crop=img.crop((max(0,x1-mg),max(0,y1-mg),
                           min(img.width,x2+mg),min(img.height,y2+mg)))
            ft=self.mtcnn_aln(crop)
            if ft is None:
                import torchvision.transforms as T
                t=T.Compose([T.Resize((160,160)),T.ToTensor(),
                             T.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])])
                ft=t(crop)
            ft=ft.unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                emb=self.resnet(ft).cpu().numpy()[0]
            n=norm(emb)
            if n<1e-9: return None
            return emb/n
        except Exception as e:
            print(f"[MOTOR] emb erro: {e}"); return None

    def obter(self):
        with self._lock: return list(self._cache),self._fps_det

    @property
    def pronto(self): return self._pronto


# ═══════════════════════════════════════════════════════════
#  APRENDIZAGEM PROGRESSIVA POR UID
#
#  Mantém o estado de aprendizagem de cada uid activo:
#  — em que fase está (1, 2, 3)
#  — quando foi o último embedding recolhido
#  — tempo total que o rosto foi visto (para determinar fase)
#
#  A cada frame onde o uid está visível, decide se está
#  na hora de recolher mais um embedding com base no
#  intervalo da fase actual.
# ═══════════════════════════════════════════════════════════
class GestorAprendizagem:
    def __init__(self):
        self._estado = {}   # uid → {fase_idx, ts_ultimo_emb, tempo_visto, total_embs}
        self._lock   = threading.Lock()

    def registar_uid(self, uid):
        with self._lock:
            if uid not in self._estado:
                self._estado[uid] = {
                    "fase_idx"    : 0,           # índice em FASES (0,1,2)
                    "ts_inicio"   : time.time(),  # quando entrou
                    "ts_ultimo"   : 0.0,          # ts do último embedding
                    "total_embs"  : 0,            # total de embeddings guardados
                }

    def deve_recolher(self, uid) -> bool:
        """
        Retorna True se é altura de recolher mais um embedding
        para este uid, com base na fase actual.
        """
        agora = time.time()
        with self._lock:
            if uid not in self._estado: return False
            e = self._estado[uid]

            # determinar fase actual pelo tempo decorrido
            tempo_visto = agora - e["ts_inicio"]
            acumulado = 0.0
            for i, (duracao, intervalo, nome) in enumerate(FASES):
                acumulado += duracao
                if tempo_visto < acumulado or i == len(FASES)-1:
                    e["fase_idx"] = i
                    break

            fase_idx = e["fase_idx"]
            _, intervalo, _ = FASES[fase_idx]

            # chegou o intervalo?
            return (agora - e["ts_ultimo"]) >= intervalo

    def registar_emb_recolhido(self, uid):
        agora = time.time()
        with self._lock:
            if uid not in self._estado: return
            e = self._estado[uid]
            e["ts_ultimo"]  = agora
            e["total_embs"] += 1

    def info(self, uid):
        """Retorna (fase_nome, total_embs, tempo_visto)."""
        agora = time.time()
        with self._lock:
            if uid not in self._estado:
                return "—", 0, 0.0
            e = self._estado[uid]
            fase_idx = e["fase_idx"]
            nome = FASES[fase_idx][2]
            return nome, e["total_embs"], agora - e["ts_inicio"]

    def remover(self, uid):
        with self._lock:
            self._estado.pop(uid, None)


# ═══════════════════════════════════════════════════════════
#  BANCO DE PESSOAS — RAM
#
#  FILOSOFIA DE SEPARAÇÃO RIGOROSA:
#  ─────────────────────────────────────────────────────────
#
#  O problema "duas pessoas com o mesmo ID" acontece quando:
#    A) LIMIAR_MATCH muito alto (0.80+) aceita matches fracos
#    B) Não existe verificação de margem entre candidatos
#    C) A zona de dúvida atribui ao melhor sem mais critério
#
#  SOLUÇÃO — 4 camadas de verificação:
#
#  1. LIMIAR mais baixo (0.75) — só aceita matches fortes
#
#  2. MARGEM DE SEPARAÇÃO (0.10)
#     O 1º candidato tem de estar ≥0.10 ACIMA do 2º.
#     Ex: uid1=0.82, uid2=0.75 → margem=0.07 < 0.10 → NOVO ID
#         uid1=0.84, uid2=0.70 → margem=0.14 ≥ 0.10 → match uid1
#     Elimina confusão entre pessoas parecidas.
#
#  3. VERIFICAÇÃO CRUZADA (cross-check)
#     Para confirmar que uid_A pertence ao embedding recebido,
#     verifica também se o embedding médio de uid_A se aproxima
#     mais deste embedding do que de qualquer outro uid.
#     Se não → ambíguo → novo ID.
#
#  4. VOTOS REFORÇADOS (3 votos, limiar estrito)
#     Na zona de dúvida exige 3 embeddings do banco acima de
#     LIMIAR_REJEICAO+0.05, não apenas do limiar base.
#
# ═══════════════════════════════════════════════════════════
class BancoPessoas:
    _next_uid = 1

    def __init__(self):
        self.bd              = {}   # uid → [{"emb": ndarray, "q": float}]
        self._ultimo_registo = 0.0
        self._lock           = threading.Lock()
        print("[BANCO] Iniciado em RAM — modo separação rigorosa.")

    def reset(self):
        with self._lock:
            self.bd.clear()
            BancoPessoas._next_uid = 1
        print("[BANCO] Resetado.")

    def identificar(self, emb, qualidade=0.9, uid_rastreador=None):
        """
        Devolve (uid, é_novo). uid==-1 → embedding inválido.

        Nunca mistura dois rostos diferentes no mesmo uid.
        Em caso de dúvida → cria ID novo.
        """
        if float(norm(emb)) < 0.5:
            return -1, False

        with self._lock:
            if not self.bd:
                return self._novo(emb, qualidade)

            # ── calcular similaridade máx para cada uid ───
            sims = {uid: _sim_max(emb, lista) for uid, lista in self.bd.items()}

            # ordenar do mais similar para o menos
            ranking = sorted(sims.items(), key=lambda x: -x[1])
            melhor_uid, melhor_sim = ranking[0]
            segundo_sim = ranking[1][1] if len(ranking) > 1 else 0.0
            margem      = melhor_sim - segundo_sim

            # ── CAMADA 1: abaixo do limiar → pessoa nova ──
            if melhor_sim < LIMIAR_REJEICAO:
                if time.time() - self._ultimo_registo >= COOLDOWN_NOVO:
                    return self._novo(emb, qualidade)
                # cooldown activo — fallback seguro: não atribuir a ninguém
                return -1, False

            # ── CAMADA 2: zona de dúvida ──────────────────
            if melhor_sim < LIMIAR_MATCH:
                # verificar hint do rastreador primeiro
                if uid_rastreador is not None and uid_rastreador in self.bd:
                    sim_hint = sims[uid_rastreador]
                    votos_hint = _votos(emb, self.bd[uid_rastreador], LIMIAR_REJEICAO + 0.05)
                    # hint aceite só se: sim alta + votos + é claramente o melhor
                    if (sim_hint >= LIMIAR_REJEICAO + 0.05
                            and votos_hint >= VOTOS_MINIMOS
                            and sim_hint >= segundo_sim + MARGEM_SEPARACAO * 0.5):
                        return uid_rastreador, False

                # sem hint — exige votos E margem mínima
                votos = _votos(emb, self.bd[melhor_uid], LIMIAR_REJEICAO + 0.05)
                if votos >= VOTOS_MINIMOS and margem >= MARGEM_SEPARACAO * 0.5:
                    return melhor_uid, False

                # dúvida sem resolução → novo ID
                if time.time() - self._ultimo_registo >= COOLDOWN_NOVO:
                    print(f"[BANCO] ❓ Dúvida sim={melhor_sim:.3f} margem={margem:.3f} → novo ID")
                    return self._novo(emb, qualidade)
                return -1, False   # cooldown → ignorar esta detecção

            # ── CAMADA 3: match seguro (sim >= LIMIAR_MATCH) ──
            # Verificar margem de separação
            if margem < MARGEM_SEPARACAO:
                # dois UIDs demasiado próximos entre si — ambíguo
                # pode acontecer com gémeos ou pessoas parecidas
                if time.time() - self._ultimo_registo >= COOLDOWN_NOVO:
                    print(f"[BANCO] ⚠ Margem insuficiente {margem:.3f} < {MARGEM_SEPARACAO} → novo ID")
                    return self._novo(emb, qualidade)
                return -1, False

            # ── CAMADA 4: cross-check ─────────────────────
            # Verificar que o embedding médio do uid vencedor
            # também aponta para este embedding (consistência bidirecional)
            media_uid = np.mean([it["emb"] for it in self.bd[melhor_uid]], axis=0)
            media_uid = media_uid / max(norm(media_uid), 1e-9)
            # calcular a que uid o centróide do vencedor se aproxima mais
            sims_media = {uid: _sim(media_uid, np.mean([it["emb"] for it in lista], axis=0))
                          for uid, lista in self.bd.items() if len(lista) > 0}
            uid_centroide = max(sims_media, key=sims_media.get)
            if uid_centroide != melhor_uid:
                # inconsistência — os centróides não concordam
                if time.time() - self._ultimo_registo >= COOLDOWN_NOVO:
                    print(f"[BANCO] ⚠ Cross-check falhou uid={melhor_uid}≠centróide={uid_centroide} → novo ID")
                    return self._novo(emb, qualidade)
                return -1, False

            # ── APROVADO em todas as camadas ──────────────
            return melhor_uid, False

    def adicionar_embedding(self, uid, emb, qualidade):
        """
        Adiciona embedding progressivo ao uid.
        Só aceita se:
          1. uid existe
          2. embedding é DIVERSO dos existentes (não duplica ângulo)
          3. NÃO fica mais próximo de outro uid do que do próprio uid
             (evita "contaminar" o perfil com embeddings ambíguos)
        """
        with self._lock:
            if uid not in self.bd:
                return False

            lista = self.bd[uid]

            # verificação de diversidade — não guardar duplicados
            if not _e_diverso(emb, lista):
                return False

            # verificação de pureza — este embedding não pode estar
            # mais próximo de outro uid do que do uid destino
            sim_proprio = _sim_max(emb, lista)
            for outro_uid, outra_lista in self.bd.items():
                if outro_uid == uid: continue
                if _sim_max(emb, outra_lista) > sim_proprio + 0.05:
                    # mais próximo de outro uid → não contaminar
                    return False

            if len(lista) >= MAX_EMB_TOTAL:
                idx_min = min(range(len(lista)), key=lambda i: lista[i]["q"])
                if qualidade > lista[idx_min]["q"]:
                    lista[idx_min] = {"emb": emb.copy(), "q": qualidade}
                    return True
                return False

            lista.append({"emb": emb.copy(), "q": qualidade})
            return True

    def num_embeddings(self, uid):
        with self._lock:
            return len(self.bd.get(uid, []))

    def debug_sims(self, emb):
        with self._lock:
            return sorted(
                [(uid, _sim_max(emb, lista), _sim_media(emb, lista))
                 for uid, lista in self.bd.items()],
                key=lambda x: -x[1])

    def _novo(self, emb, qualidade):
        uid = BancoPessoas._next_uid
        BancoPessoas._next_uid += 1
        self.bd[uid] = [{"emb": emb.copy(), "q": qualidade}]
        self._ultimo_registo = time.time()
        print(f"[BANCO] ✨ uid={uid}  total={len(self.bd)}")
        return uid, True

    @property
    def total(self): return len(self.bd)


# ═══════════════════════════════════════════════════════════
#  CONTADOR DE PRESENÇA
# ═══════════════════════════════════════════════════════════
class ContadorPresenca:
    def __init__(self):
        self.dados={}; self._lock=threading.Lock()

    def _init(self,uid):
        self.dados[uid]={"tp":0.0,"ts_seg":None,"ts_aus":None,"valido":False,"alerta":False}

    def marcar_presente(self,uid):
        agora=time.time()
        with self._lock:
            if uid not in self.dados: self._init(uid)
            d=self.dados[uid]
            if d["ts_seg"] is None: d["ts_seg"]=agora; d["ts_aus"]=None

    def marcar_ausente(self,uid):
        agora=time.time()
        with self._lock:
            if uid not in self.dados: return
            d=self.dados[uid]
            if d["ts_seg"] is not None:
                d["tp"]+=agora-d["ts_seg"]; d["ts_seg"]=None
                if d["tp"]>=TEMPO_MIN_PRESENCA: d["valido"]=True
            if d["ts_aus"] is None: d["ts_aus"]=agora

    def deve_sinalizar(self,uid):
        with self._lock:
            if uid not in self.dados: return False
            d=self.dados[uid]
            if d["alerta"]: return False
            tp=d["tp"]+(time.time()-d["ts_seg"] if d["ts_seg"] else 0)
            if tp<TEMPO_MIN_PRESENCA: return False
            if d["ts_aus"] is None: return False
            if time.time()-d["ts_aus"]>=TEMPO_AUSENCIA_SAIDA:
                d["alerta"]=True; return True
            return False

    def info(self,uid):
        agora=time.time()
        with self._lock:
            if uid not in self.dados: return 0.0,0.0,False
            d=self.dados[uid]
            tp=d["tp"]+(agora-d["ts_seg"] if d["ts_seg"] else 0)
            ta=(agora-d["ts_aus"]) if d["ts_aus"] else 0.0
            return tp,ta,d["valido"]


# ═══════════════════════════════════════════════════════════
#  RASTREADOR — integra aprendizagem progressiva
# ═══════════════════════════════════════════════════════════
def _centro(bbox):
    x,y,w,h=bbox; return float(x+w/2),float(y+h/2)

def _dist_esp(b1,b2):
    cx1,cy1=_centro(b1); cx2,cy2=_centro(b2)
    return math.sqrt((cx1-cx2)**2+(cy1-cy2)**2)

class Rastreador:
    TIMEOUT      = 5.0
    DIST_ESP_MAX = 160.0
    PESO_ESP     = 0.20

    def __init__(self, banco, motor, contador, aprendizagem):
        self.banco       = banco
        self.motor       = motor
        self.contador    = contador
        self.aprendiz    = aprendizagem
        self.activos     = {}
        self._pendentes  = {}

    def _score(self,det,info):
        s_emb=max(0.0,(_sim(det["emb"],info["emb"])-LIMIAR_REJEICAO)/
                  max(1.0-LIMIAR_REJEICAO,1e-6))
        s_esp=max(0.0,1.0-_dist_esp(det["bbox"],info["bbox"])/self.DIST_ESP_MAX) \
              if info.get("bbox") else 0.5
        return (1.0-self.PESO_ESP)*s_emb+self.PESO_ESP*s_esp

    def actualizar(self):
        resultados,_=self.motor.obter()
        agora=time.time()
        assoc=set()

        if resultados:
            uid_list=[u for u,inf in self.activos.items() if agora-inf["ts"]<self.TIMEOUT]
            usados_det=set()

            if uid_list:
                n_d,n_u=len(resultados),len(uid_list)
                mat=np.zeros((n_d,n_u))
                for i,r in enumerate(resultados):
                    for j,u in enumerate(uid_list):
                        mat[i,j]=self._score(r,self.activos[u])
                pares=sorted(((i,j) for i in range(n_d) for j in range(n_u)),
                             key=lambda p:-mat[p[0],p[1]])
                usados_j=set()
                for i,j in pares:
                    if mat[i,j]<0.15: break
                    if i in usados_det or j in usados_j: continue
                    uid=uid_list[j]
                    if _sim(resultados[i]["emb"],self.activos[uid]["emb"])<LIMIAR_REJEICAO: continue
                    self.activos[uid].update({
                        "bbox":resultados[i]["bbox"],"emb":resultados[i]["emb"],
                        "ts":agora,"q":resultados[i]["q"],"conf":resultados[i]["conf"]})
                    assoc.add(uid); usados_det.add(i); usados_j.add(j)

                    # ── APRENDIZAGEM PROGRESSIVA ──────────────
                    # uid já confirmado e visível → verificar se deve recolher embedding
                    if self.aprendiz.deve_recolher(uid):
                        adicionado = self.banco.adicionar_embedding(
                            uid, resultados[i]["emb"], resultados[i]["q"])
                        if adicionado:
                            self.aprendiz.registar_emb_recolhido(uid)
                            fase,total,_ = self.aprendiz.info(uid)
                            print(f"[APRENDIZ] uid={uid} +1 emb  total={total}  fase={fase}")

            for i,r in enumerate(resultados):
                if i in usados_det: continue
                hint=uid_list[0] if len(uid_list)==1 else None
                uid,is_new=self.banco.identificar(r["emb"],qualidade=r["q"],uid_rastreador=hint)
                if uid==-1: continue

                if uid not in self.activos:
                    p=self._pendentes.get(uid)
                    if p is None:
                        self._pendentes[uid]={"n":1,"bbox":r["bbox"],"emb":r["emb"],
                                              "q":r["q"],"conf":r["conf"],"ts":agora}
                    else:
                        p["n"]+=1; p.update({"bbox":r["bbox"],"emb":r["emb"],"ts":agora})
                    if self._pendentes[uid]["n"]>=FRAMES_CONF:
                        self.activos[uid]={"bbox":r["bbox"],"emb":r["emb"],
                                           "ts":agora,"q":r["q"],"conf":r["conf"]}
                        del self._pendentes[uid]; assoc.add(uid)
                        # iniciar gestão de aprendizagem para este uid
                        self.aprendiz.registar_uid(uid)
                        print(f"[RASTR] ✅ uid={uid} confirmado — aprendizagem iniciada")
                else:
                    self.activos[uid].update({"bbox":r["bbox"],"emb":r["emb"],
                                              "ts":agora,"q":r["q"],"conf":r["conf"]})
                    assoc.add(uid)
                    if self.aprendiz.deve_recolher(uid):
                        adicionado=self.banco.adicionar_embedding(uid,r["emb"],r["q"])
                        if adicionado:
                            self.aprendiz.registar_emb_recolhido(uid)
                            fase,total,_=self.aprendiz.info(uid)
                            print(f"[APRENDIZ] uid={uid} +1 emb  total={total}  fase={fase}")

        for uid in [u for u,p in list(self._pendentes.items()) if agora-p["ts"]>4.0]:
            del self._pendentes[uid]

        for uid in set(self.activos):
            if uid in assoc: self.contador.marcar_presente(uid)
            elif agora-self.activos[uid]["ts"]>self.TIMEOUT:
                self.contador.marcar_ausente(uid)

        return {uid:inf["bbox"] for uid,inf in self.activos.items()
                if agora-inf["ts"]<self.TIMEOUT}

    def ausentes_para_descida(self):
        return [uid for uid in list(self.activos) if self.contador.deve_sinalizar(uid)]


# ═══════════════════════════════════════════════════════════
#  APLICAÇÃO PRINCIPAL
# ═══════════════════════════════════════════════════════════
class TaxiApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("🚕  Gestão de Táxi – FaceNet + Aprendizagem Progressiva")
        self.geometry("1300x840")
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.banco       = BancoPessoas()
        self.contador    = ContadorPresenca()
        self.motor       = MotorFacial()
        self.aprendizagem = GestorAprendizagem()
        self.rastreador  = Rastreador(self.banco, self.motor,
                                      self.contador, self.aprendizagem)

        self.taxi_em_movimento = False
        self.total_cobrado = 0
        self.passageiros   = {}
        self.frame_count   = 0
        self.fundo_modo    = FUNDO_MODO
        self.cap           = None
        self.running       = False

        self.records_file = "viagens.json"
        self.viagens      = self._load_json(self.records_file)

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_closing)

    def _load_json(self,p):
        if os.path.exists(p):
            try:
                with open(p,"r",encoding="utf-8") as f: return json.load(f)
            except: pass
        return {}

    def _save_json(self):
        try:
            with open(self.records_file,"w",encoding="utf-8") as f:
                json.dump(self.viagens,f,ensure_ascii=False,indent=4)
        except Exception as e: print(f"[ERRO] {e}")

    # ── UI ────────────────────────────────────────────────
    def _build_ui(self):
        top=ctk.CTkFrame(self); top.pack(fill="x",padx=10,pady=6)

        ctk.CTkLabel(top,text="IP:").pack(side="left",padx=(6,2))
        self.entry_ip=ctk.CTkEntry(top,width=150)
        self.entry_ip.insert(0,"192.168.42.129"); self.entry_ip.pack(side="left",padx=4)
        ctk.CTkLabel(top,text="Porta:").pack(side="left",padx=(4,2))
        self.entry_porta=ctk.CTkEntry(top,width=70)
        self.entry_porta.insert(0,"4747"); self.entry_porta.pack(side="left",padx=4)

        self.btn_start=ctk.CTkButton(top,text="▶ Iniciar",width=100,command=self.start_capture)
        self.btn_start.pack(side="left",padx=6)
        self.btn_stop=ctk.CTkButton(top,text="■ Parar",width=100,fg_color="#c0392b",
                                    hover_color="#922b21",command=self.stop_capture,state="disabled")
        self.btn_stop.pack(side="left",padx=4)
        ctk.CTkButton(top,text="📋 Relatório",width=110,
                      command=self.mostrar_relatorio).pack(side="left",padx=4)

        ctk.CTkLabel(top,text="Fundo:",font=("Arial",11)).pack(side="left",padx=(10,2))
        for modo,emoji,cor,cor2 in [("blur","🌫 Blur","#1a6b3c","#145530"),
                                     ("dark","🌑 Dark","#2c3e50","#1a252f"),
                                     ("bw","⬛ P&B","#4a4a4a","#333333"),
                                     ("off","❌ Off","#7f8c8d","#636e72")]:
            ctk.CTkButton(top,text=emoji,width=75,fg_color=cor,hover_color=cor2,
                          command=lambda m=modo:self._set_fundo(m)).pack(side="left",padx=2)
        self.lbl_fundo=ctk.CTkLabel(top,text="● off",font=("Arial",11,"bold"),text_color="#e74c3c")
        self.lbl_fundo.pack(side="left",padx=4)
        ctk.CTkButton(top,text="🗑 Reset",width=90,fg_color="#7f8c8d",hover_color="#636e72",
                      command=self._reset).pack(side="left",padx=4)
        self.lbl_bd=ctk.CTkLabel(top,text="RAM: 0 p.",font=("Arial",11),text_color="#95a5a6")
        self.lbl_bd.pack(side="left",padx=6)

        gps=ctk.CTkFrame(self); gps.pack(fill="x",padx=10,pady=4)
        for lbl,attr,default,width in [
            ("Lat:","entry_lat","-8.8383",110),("Lon:","entry_lon","13.2344",110),
            ("Vel km/h:","entry_vel","0",60)]:
            ctk.CTkLabel(gps,text=lbl).pack(side="left",padx=(6,2))
            e=ctk.CTkEntry(gps,width=width); e.insert(0,default)
            e.pack(side="left",padx=4); setattr(self,attr,e)
        self.lbl_mov=ctk.CTkLabel(gps,text="🔴 Parado",font=("Arial",13,"bold"))
        self.lbl_mov.pack(side="left",padx=12)

        mid=ctk.CTkFrame(self); mid.pack(fill="both",expand=True,padx=10,pady=4)
        self.video_label=ctk.CTkLabel(mid,
            text="⏳ A carregar MTCNN + FaceNet...\nAguarde ~10s",
            width=700,height=430,font=("Arial",14))
        self.video_label.pack(side="left",padx=6,pady=6)

        painel=ctk.CTkFrame(mid)
        painel.pack(side="left",fill="both",expand=True,padx=6,pady=6)
        ctk.CTkLabel(painel,text="PAINEL",font=("Arial",15,"bold")).pack(pady=(10,4))

        fu=ctk.CTkFrame(painel,fg_color="#0d2b45",corner_radius=14)
        fu.pack(fill="x",padx=10,pady=4)
        ctk.CTkLabel(fu,text="PESSOAS ÚNICAS",font=("Arial",11,"bold"),
                     text_color="#6ab0e0").pack(pady=(8,2))
        self.lbl_unicas=ctk.CTkLabel(fu,text="0",font=("Arial",52,"bold"),text_color="#fff")
        self.lbl_unicas.pack()
        ctk.CTkLabel(fu,text="FaceNet + Aprendizagem Progressiva",
                     font=("Arial",9),text_color="#557a99").pack(pady=(0,6))

        self.lbl_ativos  = ctk.CTkLabel(painel,text="No táxi: 0",font=("Arial",12))
        self.lbl_ativos.pack(pady=1)
        self.lbl_viagens = ctk.CTkLabel(painel,text="Viagens: 0",font=("Arial",12))
        self.lbl_viagens.pack(pady=1)
        self.lbl_money   = ctk.CTkLabel(painel,text="Total: 0 kz",
                                        font=("Arial",16,"bold"),text_color="#2ecc71")
        self.lbl_money.pack(pady=4)
        self.lbl_det     = ctk.CTkLabel(painel,text="🔍 Aguardando...",font=("Arial",11))
        self.lbl_det.pack(pady=1)
        self.lbl_fps     = ctk.CTkLabel(painel,text="FPS: --",font=("Arial",10),text_color="#888")
        self.lbl_fps.pack(pady=1)

        # ── painel de aprendizagem ─────────────────────────
        ctk.CTkLabel(painel,text="📈 Aprendizagem por passageiro:",
                     font=("Arial",10,"bold")).pack(anchor="w",padx=6,pady=(8,0))
        self.frame_aprendiz=ctk.CTkScrollableFrame(painel,height=110)
        self.frame_aprendiz.pack(fill="x",padx=6,pady=2)
        self._aprendiz_labels={}

        ctk.CTkLabel(painel,text="🟡 Aguardando  🟢 Em viagem  🔴 Saiu",
                     font=("Arial",10)).pack(pady=2)
        ctk.CTkLabel(painel,text="Log:",font=("Arial",11)).pack(anchor="w",padx=6,pady=(4,0))
        self.log_box=ctk.CTkTextbox(painel,height=140,wrap="word",font=("Courier",10))
        self.log_box.pack(fill="both",expand=True,padx=4,pady=4)

        self.lbl_status=ctk.CTkLabel(self,text="A carregar FaceNet...",anchor="w")
        self.lbl_status.pack(fill="x",padx=14,pady=4)

    def _log(self,msg):
        ts=datetime.now().strftime("%H:%M:%S")
        self.log_box.insert("end",f"[{ts}] {msg}\n")
        self.log_box.see("end")

    def _reset(self):
        self.banco.reset()
        self.lbl_unicas.configure(text="0")
        self.lbl_bd.configure(text="RAM: 0 p.")
        self._log("🗑 Banco resetado.")

    def _set_fundo(self,modo):
        self.fundo_modo=modo
        cores={"blur":"#2ecc71","dark":"#3498db","bw":"#95a5a6","off":"#e74c3c"}
        nomes={"blur":"● blur","dark":"● dark","bw":"● p&b","off":"● off"}
        self.lbl_fundo.configure(text=nomes.get(modo,modo),text_color=cores.get(modo,"#fff"))

    # ── câmera ────────────────────────────────────────────
    def start_capture(self):
        if not self.motor.pronto:
            self.lbl_status.configure(text="⏳ Aguardando FaceNet...")
            self.after(1500,self.start_capture); return
        url=f"http://{self.entry_ip.get().strip()}:{self.entry_porta.get().strip()}/video"
        self.cap=cv2.VideoCapture(url)
        if not self.cap.isOpened():
            self.lbl_status.configure(text="❌ Stream não disponível."); return
        self.running=True
        self.btn_start.configure(state="disabled")
        self.btn_stop.configure(state="normal")
        self.lbl_status.configure(text=f"✅ FaceNet + Aprendizagem Progressiva · {url}")
        threading.Thread(target=self._loop,daemon=True).start()
        self._poll_gps()
        self._poll_aprendizagem()

    def stop_capture(self):
        self.running=False
        self.btn_start.configure(state="normal")
        self.btn_stop.configure(state="disabled")

    def _poll_gps(self):
        if not self.running: return
        try: vel=float(self.entry_vel.get())
        except: vel=0.0
        self.taxi_em_movimento=vel>=VELOCIDADE_MINIMA
        self.lbl_mov.configure(text=f"🟢 {vel:.0f} km/h" if self.taxi_em_movimento else "🔴 Parado")
        self.after(2000,self._poll_gps)

    def _get_gps(self):
        try: return float(self.entry_lat.get()),float(self.entry_lon.get())
        except: return -8.8383,13.2344

    def _poll_aprendizagem(self):
        """Actualiza painel de aprendizagem progressiva (1x/seg)."""
        if not self.running: return

        uids_activos={uid for uid,p in self.passageiros.items()
                      if p["estado"] in ("aguardando","em_viagem")}

        for uid in list(self._aprendiz_labels):
            if uid not in uids_activos:
                self._aprendiz_labels[uid].destroy()
                del self._aprendiz_labels[uid]

        for uid in uids_activos:
            fase,total_embs,tempo_visto=self.aprendizagem.info(uid)
            n_embs=self.banco.num_embeddings(uid)
            tp,ta,valido=self.contador.info(uid)
            estado=self.passageiros[uid]["estado"]

            # barra de embeddings (0–MAX_EMB_TOTAL)
            prog=min(1.0, n_embs/MAX_EMB_TOTAL)
            barras=int(prog*12)
            barra="█"*barras+"░"*(12-barras)

            # icone de fase
            if   fase=="ENTRADA":        icone_fase="🟡"
            elif fase=="CONSOLIDAÇÃO":   icone_fase="🔵"
            else:                        icone_fase="🟢"

            # icone de estado do passageiro
            if   estado=="em_viagem":  icone_p="🚗"
            elif valido:               icone_p="✅"
            else:                      icone_p="⏳"

            txt=(f"{icone_p} P{uid} {icone_fase}{fase[:4]}  "
                 f"emb:{n_embs:>2}/{MAX_EMB_TOTAL}[{barra}]  "
                 f"vis:{tempo_visto:.0f}s  pres:{tp:.0f}s")

            if uid not in self._aprendiz_labels:
                lbl=ctk.CTkLabel(self.frame_aprendiz,text=txt,
                                 font=("Courier",10),anchor="w")
                lbl.pack(fill="x",padx=4,pady=1)
                self._aprendiz_labels[uid]=lbl
            else:
                self._aprendiz_labels[uid].configure(text=txt)

        self.after(1000,self._poll_aprendizagem)

    # ── loop principal ────────────────────────────────────
    def _loop(self):
        fc=0
        while self.running:
            ret,frame=self.cap.read()
            if not ret: time.sleep(0.1); continue
            frame=cv2.resize(frame,(700,430))
            fc+=1
            if fc%2==0: self.motor.enviar_frame(frame)

            assoc=self.rastreador.actualizar()
            for uid in assoc:
                if uid not in self.passageiros: self._entrou(uid)
            if self.taxi_em_movimento:
                for uid,p in self.passageiros.items():
                    if p["estado"]=="aguardando":
                        p["estado"]="em_viagem"
                        p["entrada_lat"],p["entrada_lon"]=self._get_gps()
                        self.after(0,self._log,f"P{uid} → em viagem!")
            for uid in self.rastreador.ausentes_para_descida():
                self._sinalizar_descida(uid)

            brilho=float(np.mean(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)))
            if self.fundo_modo!="off" and assoc:
                frame=aplicar_fundo(frame,list(assoc.values()),self.fundo_modo)
            self._desenhar(frame,assoc)
            resultados,fps_det=self.motor.obter()
            self.after(0,self._update_ui,frame,len(resultados),fps_det,brilho)
            time.sleep(0.033)
        if self.cap: self.cap.release()

    def _entrou(self,uid):
        lat,lon=self._get_gps()
        self.passageiros[uid]={
            "id":uid,"estado":"aguardando",
            "entrada_ts":datetime.now().isoformat(),
            "entrada_lat":lat,"entrada_lon":lon,
            "saida_ts":None,"saida_lat":None,"saida_lon":None,
            "distancia_km":0.0,"valor":0}
        self.after(0,self._log,f"✅ P{uid} entrou  (total:{self.banco.total})")

    def _sinalizar_descida(self,uid):
        p=self.passageiros.get(uid)
        if p is None or p["estado"]=="concluido": return
        lat,lon=self._get_gps()
        p.update({"saida_ts":datetime.now().isoformat(),"saida_lat":lat,"saida_lon":lon,
                  "estado":"concluido","valor":VALOR_DESCIDA})
        self.total_cobrado+=VALOR_DESCIDA
        vid=f"v{uid}_{datetime.now().strftime('%H%M%S')}"
        self.viagens[vid]=p.copy(); self._save_json()
        tp,_,_=self.contador.info(uid)
        fase,n_embs,_=self.aprendizagem.info(uid)
        self.after(0,self._log,
                   f"🔴 P{uid} DESCEU ({tp:.0f}s) embs:{n_embs} → {VALOR_DESCIDA}kz")
        self.after(0,self._mostrar_alerta,uid)

    def _mostrar_alerta(self,uid):
        win=ctk.CTkToplevel(self); win.title("🔴 Descida"); win.geometry("320x150"); win.grab_set()
        ctk.CTkLabel(win,text=f"🔴  P{uid} desceu!",font=("Arial",18,"bold"),
                     text_color="#e74c3c").pack(pady=(20,8))
        ctk.CTkLabel(win,text=f"Cobrado: {VALOR_DESCIDA} kz",font=("Arial",15),
                     text_color="#2ecc71").pack(pady=4)
        ctk.CTkButton(win,text="OK",width=100,command=win.destroy).pack(pady=12)

    def _desenhar(self,frame,assoc):
        COR={"aguardando":(0,200,255),"em_viagem":(0,220,0),"concluido":(160,160,160)}
        FASES_COR={"ENTRADA":(255,200,0),"CONSOLIDAÇÃO":(0,180,255),"FORTALECIMENTO":(0,255,120)}

        for uid,bbox in assoc.items():
            x,y,w,h=bbox
            estado=self.passageiros.get(uid,{}).get("estado","aguardando")
            cor=COR.get(estado,(200,200,200))
            inf=self.rastreador.activos.get(uid,{})
            conf=inf.get("conf",0)
            tp,ta,_=self.contador.info(uid)
            fase,n_embs,tempo_visto=self.aprendizagem.info(uid)

            espessura = 1 if fase=="ENTRADA" else 2 if fase=="CONSOLIDAÇÃO" else 3
            cv2.rectangle(frame,(x,y),(x+w,y+h),cor,espessura)

            # barra de aprendizagem
            prog=min(1.0,n_embs/MAX_EMB_TOTAL)
            barra_w=int(w*prog)
            cor_fase=FASES_COR.get(fase,(200,200,200))
            cv2.rectangle(frame,(x,y-5),(x+barra_w,y),cor_fase,-1)
            cv2.rectangle(frame,(x,y-5),(x+w,y),(80,80,80),1)

            # calcular margem de separação para debug visual
            emb_uid=inf.get("emb",np.zeros(512,dtype=np.float32))
            sims=self.banco.debug_sims(emb_uid)
            margem_str=""
            cor_margem=(200,200,200)
            if len(sims)>=2:
                margem=sims[0][1]-sims[1][1]
                margem_str=f" Δ{margem:.2f}"
                # verde se margem boa, amarelo se ok, vermelho se perigosa
                if   margem >= MARGEM_SEPARACAO:         cor_margem=(0,220,0)
                elif margem >= MARGEM_SEPARACAO*0.5:     cor_margem=(0,180,255)
                else:                                     cor_margem=(0,80,255)

            sim_str=f"{sims[0][1]:.2f}" if sims else "?"
            l1=f"P{uid} sim:{sim_str}{margem_str} emb:{n_embs}"
            l2=f"{fase[:4]} vis:{tempo_visto:.0f}s pres:{tp:.0f}s"
            if ta>3: l2+=f" aus:{ta:.0f}s"

            cv2.rectangle(frame,(x,max(y-50,0)),(x+w,max(y-5,0)),(0,0,0),-1)
            cv2.putText(frame,l1,(x+3,max(y-36,10)),
                        cv2.FONT_HERSHEY_SIMPLEX,0.40,cor,1,cv2.LINE_AA)
            cv2.putText(frame,l2,(x+3,max(y-20,24)),
                        cv2.FONT_HERSHEY_SIMPLEX,0.38,cor_fase,1,cv2.LINE_AA)

        cv2.rectangle(frame,(0,0),(330,52),(0,0,0),-1)
        cv2.putText(frame,f"Pessoas unicas: {self.banco.total}",
                    (8,34),cv2.FONT_HERSHEY_SIMPLEX,0.90,(255,255,255),2,cv2.LINE_AA)

    def _update_ui(self,frame,n,fps_det,brilho):
        img_tk=ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)))
        self.video_label.configure(image=img_tk); self.video_label.image=img_tk
        ativos=[(uid,p) for uid,p in self.passageiros.items()
                if p["estado"] in ("aguardando","em_viagem")]
        self.lbl_unicas.configure(text=str(self.banco.total))
        self.lbl_ativos.configure(
            text=f"No táxi: {len(ativos)} — {', '.join(f'P{u}' for u,_ in ativos)}"
            if ativos else "No táxi: 0")
        self.lbl_viagens.configure(text=f"Viagens: {len(self.viagens)}")
        self.lbl_money.configure(text=f"Total: {self.total_cobrado:,} kz")
        self.lbl_fps.configure(text=f"FPS: {fps_det}")
        self.lbl_bd.configure(text=f"RAM: {self.banco.total} p.")
        self.lbl_det.configure(
            text=f"🟢 {n} rosto(s)" if n>0 else "🔴 Sem rosto",
            text_color="#2ecc71" if n>0 else "#e74c3c")

    def mostrar_relatorio(self):
        win=ctk.CTkToplevel(self); win.title("📋 Relatório"); win.geometry("700x480")
        txt=ctk.CTkTextbox(win,wrap="none",font=("Courier",12))
        txt.pack(fill="both",expand=True,padx=10,pady=10)
        linhas=["="*70,
                f"  RELATÓRIO — {datetime.now().strftime('%d/%m/%Y %H:%M')}",
                f"  Motor: FaceNet VGGFace2 + MTCNN + Aprendizagem Progressiva",
                f"  Pessoas: {self.banco.total}  Match:{LIMIAR_MATCH}  Rejeição:{LIMIAR_REJEICAO}",
                "="*70,
                f"{'ID':<5} {'Entrada':<20} {'Saída':<20} {'kz':<10}","-"*70]
        total=0
        for v in self.viagens.values():
            linhas.append(f"{v['id']:<5} "
                          f"{str(v.get('entrada_ts',''))[:19]:<20} "
                          f"{str(v.get('saida_ts',''))[:19]:<20} "
                          f"{v.get('valor',0):<10}")
            total+=v.get("valor",0)
        linhas+=["-"*70,f"  TOTAL: {total:,} kz","="*70]
        txt.insert("end","\n".join(linhas))

    def _on_closing(self):
        self._save_json(); self.running=False
        time.sleep(0.2)
        if self.cap: self.cap.release()
        gc.collect(); self.destroy()


if __name__ == "__main__":
    app = TaxiApp()
    app.mainloop()