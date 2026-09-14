import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import skfuzzy as fuzzy
from skfuzzy import control as ctrl
from ultralytics import YOLO

LIMIAR_RISCO = 42.5
IMG_SIZE = 640
CONF_DETEC = 0.25
CONF_POSE = 0.25


def criar_motor_fuzzy():
    distancia = ctrl.Antecedent(np.arange(0, 1000, 1), "distancia")
    velocidade = ctrl.Antecedent(np.arange(0, 1000, 1), "velocidade")
    risco = ctrl.Consequent(np.arange(0, 101, 1), "risco")

    distancia["perto"] = fuzzy.trapmf(distancia.universe, [0, 0, 80, 180])
    distancia["media"] = fuzzy.trimf(distancia.universe, [80, 250, 400])
    distancia["longe"] = fuzzy.trapmf(distancia.universe, [300, 500, 1000, 1000])

    velocidade["lenta"] = fuzzy.trapmf(velocidade.universe, [0, 0, 80, 180])
    velocidade["moderada"] = fuzzy.trimf(velocidade.universe, [80, 250, 500])
    velocidade["rapida"] = fuzzy.trapmf(velocidade.universe, [300, 500, 1000, 1000])

    risco["baixo"] = fuzzy.trapmf(risco.universe, [0, 0, 25, 50])
    risco["medio"] = fuzzy.trimf(risco.universe, [25, 50, 75])
    risco["alto"] = fuzzy.trapmf(risco.universe, [50, 75, 100, 100])

    regras = [
        ctrl.Rule(distancia["perto"] & velocidade["rapida"], risco["alto"]),
        ctrl.Rule(distancia["perto"] & velocidade["moderada"], risco["alto"]),
        ctrl.Rule(distancia["perto"] & velocidade["lenta"], risco["medio"]),
        ctrl.Rule(distancia["media"] & velocidade["rapida"], risco["alto"]),
        ctrl.Rule(distancia["media"] & velocidade["moderada"], risco["medio"]),
        ctrl.Rule(distancia["media"] & velocidade["lenta"], risco["baixo"]),
        ctrl.Rule(distancia["longe"], risco["baixo"]),
    ]

    sistema = ctrl.ControlSystem(regras)
    return ctrl.ControlSystemSimulation(sistema)


def localizar_modelo_objetos():
    candidatos = [
        Path("runs/detect/treino_vigilia/weights/best.pt"),
        Path("runs/detect/treino_vigilia/weights/last.pt"),
        Path("yolov8n.pt"),
    ]
    for caminho in candidatos:
        if caminho.exists():
            return str(caminho)
    return "yolov8n.pt"


def centro_caixa(box):
    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(float)
    return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=np.float32)


def nomes_de_classes(resultado):
    return getattr(resultado, "names", {}) or {}


def obter_deteccoes(resultado):
    pessoas = []
    objetos = []
    if resultado.boxes is None:
        return pessoas, objetos

    nomes = nomes_de_classes(resultado)
    for box in resultado.boxes:
        classe_idx = int(box.cls.item())
        confianca = float(box.conf.item())
        if confianca < CONF_DETEC:
            continue

        nome = nomes.get(classe_idx, str(classe_idx)).lower()
        centro = centro_caixa(box)
        item = {"box": box, "id": classe_idx, "nome": nome, "centro": centro, "conf": confianca}

        if "person" in nome:
            pessoas.append(item)
        elif any(token in nome for token in ["bag", "backpack", "handbag", "suitcase", "luggage", "purse"]):
            objetos.append(item)

    return pessoas, objetos


def calcular_risco(simulador, distancia, velocidade):
    simulador.input["distancia"] = float(np.clip(distancia, 0, 999))
    simulador.input["velocidade"] = float(np.clip(velocidade, 0, 1000))
    try:
        simulador.compute()
        return float(simulador.output["risco"])
    except (KeyError, ValueError, RuntimeError):
        return 0.0


def get_mao_mais_proxima(pontos, confiancas, centro_mochila):
    maos = []
    for indice in (9, 10):
        if indice < len(pontos) and confiancas[indice] >= CONF_POSE:
            maos.append(np.array(pontos[indice][:2], dtype=np.float32))

    if not maos:
        return None
    return min(maos, key=lambda ponto: np.linalg.norm(ponto - centro_mochila))


def reproduzir_video(caminho_video, largura_janela=1024, altura_janela=768):
    print(f"\nReproduzindo vídeo de resultado: {caminho_video} (Pressione 'q' ou ESC para fechar)...")
    cap = cv2.VideoCapture(caminho_video)
    if not cap.isOpened():
        print(f"ERRO: Não foi possível abrir o vídeo para reprodução: {caminho_video}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    delay = max(1, int(1000 / fps))
    janela = "Vigil-IA - Resultado"
    cv2.namedWindow(janela, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(janela, largura_janela, altura_janela)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow(janela, frame)
        tecla = cv2.waitKey(delay) & 0xFF
        if tecla in (ord("q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()


def processar_video_pose(video_entrada, video_saida):
    print("=== INICIANDO DETECÇÃO COM POSE ESTIMATION E ESQUELETO ===")

    modelo_objetos = YOLO(localizar_modelo_objetos())
    modelo_pose = YOLO("yolov8n-pose.pt")
    simulador_fuzzy = criar_motor_fuzzy()

    cap = cv2.VideoCapture(video_entrada)
    if not cap.isOpened():
        print(f"ERRO: Não foi possível abrir o vídeo: {video_entrada}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(video_saida, fourcc, fps, (width, height))

    posicao_mao_anterior = None
    risco_maximo = 0.0
    tipo_alerta = "NORMAL"

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results_obj = modelo_objetos(frame, imgsz=IMG_SIZE, conf=CONF_DETEC, verbose=False)[0]
        results_pose = modelo_pose(frame, imgsz=IMG_SIZE, conf=CONF_POSE, verbose=False)[0]

        annotated = frame.copy()
        pessoas, bolsas = obter_deteccoes(results_obj)

        if results_pose.keypoints is not None and len(results_pose.keypoints) > 0:
            keypoints = results_pose.keypoints.xy[0].cpu().numpy()
            confs = results_pose.keypoints.conf[0].cpu().numpy()

            for pessoa in pessoas:
                pessoa_centro = pessoa["centro"]
                cv2.circle(annotated, (int(pessoa_centro[0]), int(pessoa_centro[1])), 5, (255, 0, 0), -1)

            for bolsa in bolsas:
                centro_bolsa = bolsa["centro"]
                cv2.rectangle(
                    annotated,
                    (int(centro_bolsa[0]) - 10, int(centro_bolsa[1]) - 10),
                    (int(centro_bolsa[0]) + 10, int(centro_bolsa[1]) + 10),
                    (255, 165, 0),
                    2,
                )
                cv2.putText(
                    annotated,
                    "Bolsa",
                    (int(centro_bolsa[0]) + 10, int(centro_bolsa[1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 165, 0),
                    2,
                )

                for pessoa in pessoas:
                    centro_pessoa = pessoa["centro"]
                    mao = get_mao_mais_proxima(keypoints, confs, centro_bolsa)
                    if mao is None:
                        continue

                    mao = np.array(mao, dtype=np.float32)
                    distancia = float(np.linalg.norm(mao - centro_bolsa))
                    if posicao_mao_anterior is not None:
                        velocidade = float(np.linalg.norm(mao - posicao_mao_anterior))
                    else:
                        velocidade = 0.0
                    posicao_mao_anterior = mao

                    risco = calcular_risco(simulador_fuzzy, distancia, velocidade)
                    risco_maximo = max(risco_maximo, risco)

                    cv2.line(annotated, (int(mao[0]), int(mao[1])), (int(centro_bolsa[0]), int(centro_bolsa[1])), (0, 255, 255), 2)
                    cv2.circle(annotated, (int(mao[0]), int(mao[1])), 6, (0, 255, 255), -1)

                    if risco >= LIMIAR_RISCO:
                        tipo_alerta = "SUSPEITA: MÃO NA BOLSA"
                        cv2.putText(
                            annotated,
                            "ALERTA",
                            (max(10, int(centro_bolsa[0]) - 20), max(30, int(centro_bolsa[1]) - 20)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 0, 255),
                            2,
                        )
                    else:
                        tipo_alerta = "NORMAL"

        if risco_maximo >= LIMIAR_RISCO:
            cor_status = (0, 0, 255)
            texto_status = f"ALERTA ({tipo_alerta}): {risco_maximo:.1f}%"
        else:
            cor_status = (0, 255, 0)
            texto_status = f"STATUS: NORMAL ({risco_maximo:.1f}%)"

        cv2.rectangle(annotated, (0, 0), (width, 50), (0, 0, 0), -1)
        cv2.putText(annotated, texto_status, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, cor_status, 2)
        out.write(annotated)

    cap.release()
    out.release()
    print(f"\nVídeo com rastreamento de pose salvo em: {video_saida}")
    reproduzir_video(video_saida)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validação de pose e risco de vigília em vídeo.")
    parser.add_argument("--input", default="video_teste.mp4", help="Caminho do vídeo de entrada.")
    parser.add_argument("--output", default="resultado_vigilia_pose.mp4", help="Caminho do vídeo de saída.")
    args = parser.parse_args()

    if os.path.exists(args.input):
        processar_video_pose(args.input, args.output)
    else:
        print(f"Coloque o arquivo '{args.input}' na pasta do projeto e execute novamente.")