import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import skfuzzy as fuzzy
from skfuzzy import control as ctrl
from ultralytics import YOLO


LIMIAR_RISCO_PADRAO = 42.5
PROPORCAO_SUSPEITA_PADRAO = 0.20
IMG_SIZE = 640
CONFIANCA_DETECCAO = 0.20
CONFIANCA_POSE = 0.20
DISTANCIA_INTERACAO_NORMALIZADA = 0.90
VELOCIDADE_MINIMA_INTERACAO = 8.0


def criar_motor_fuzzy():
    distancia = ctrl.Antecedent(np.arange(0, 1000, 1), "distancia")
    velocidade = ctrl.Antecedent(np.arange(0, 1001, 1), "velocidade")
    risco = ctrl.Consequent(np.arange(0, 101, 1), "risco")

    distancia["perto"] = fuzzy.trapmf(distancia.universe, [0, 0, 100, 200])
    distancia["media"] = fuzzy.trimf(distancia.universe, [100, 250, 400])
    distancia["longe"] = fuzzy.trapmf(distancia.universe, [450, 600, 999, 999])

    velocidade["lenta"] = fuzzy.trapmf(velocidade.universe, [0, 0, 100, 250])
    velocidade["moderada"] = fuzzy.trimf(velocidade.universe, [100, 300, 500])
    velocidade["rapida"] = fuzzy.trapmf(velocidade.universe, [500, 700, 1000, 1000])

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
    return ctrl.ControlSystemSimulation(ctrl.ControlSystem(regras))


def localizar_modelo():
    candidatos = [
        #Path("runs/detect/treino_vigilia/weights/best.pt"),
        Path("runs/detect/treino_vigilia_M4_revisado/weights/best.pt"),
    ]
    for caminho in candidatos:
        if caminho.exists():
            return str(caminho)
    raise FileNotFoundError("best.pt não encontrado nos diretórios de treinamento.")


def centro_caixa(box):
    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
    return np.array([(x1 + x2) / 2, (y1 + y2) / 2], dtype=np.float32)


def obter_deteccoes(resultado):
    pessoas = []
    mochilas = []
    if resultado.boxes is None:
        return pessoas, mochilas

    for box in resultado.boxes:
        classe = int(box.cls.item())
        confianca = float(box.conf.item())
        if confianca < CONFIANCA_DETECCAO:
            continue
        if classe == 0:
            pessoas.append(box)
        elif classe == 1:
            mochilas.append(box)
    return pessoas, mochilas


def obter_esqueletos(resultado_pose):
    if resultado_pose.keypoints is None or len(resultado_pose.keypoints) == 0:
        return []

    pontos = resultado_pose.keypoints.xy.cpu().numpy()
    confiancas = resultado_pose.keypoints.conf.cpu().numpy()
    return list(zip(pontos, confiancas))


def obter_punhos(esqueleto):
    pontos, confiancas = esqueleto
    return [pontos[indice] for indice in (9, 10) if confiancas[indice] >= CONFIANCA_POSE]


def centro_esqueleto(esqueleto):
    pontos, confiancas = esqueleto
    indices = [5, 6, 11, 12]
    validos = [pontos[i] for i in indices if confiancas[i] >= CONFIANCA_POSE]
    return np.mean(validos, axis=0) if validos else None


def distancia_esqueleto_pessoa(esqueleto, centro_pessoa):
    centro = centro_esqueleto(esqueleto)
    return np.linalg.norm(centro - centro_pessoa) if centro is not None else float("inf")


def escolher_punho(punhos, centro_mochila):
    if not punhos:
        return None
    return min(punhos, key=lambda ponto: np.linalg.norm(ponto - centro_mochila))


def associar_ids(pessoas, centros_anteriores, proximo_id):
    centros_atuais = {}
    usados = set()
    for pessoa in pessoas:
        centro = centro_caixa(pessoa)
        candidatos = [
            (distancia, pessoa_id)
            for pessoa_id, centro_anterior in centros_anteriores.items()
            if pessoa_id not in usados
            for distancia in [np.linalg.norm(centro - centro_anterior)]
            if distancia < 100
        ]
        if candidatos:
            _, pessoa_id = min(candidatos)
        else:
            pessoa_id = proximo_id
            proximo_id += 1
        usados.add(pessoa_id)
        centros_atuais[pessoa_id] = centro
    return centros_atuais, proximo_id


def calcular_risco(simulador, distancia, velocidade):
    simulador.input["distancia"] = float(np.clip(distancia, 0, 999))
    simulador.input["velocidade"] = float(np.clip(velocidade, 0, 1000))
    try:
        simulador.compute()
        return float(simulador.output["risco"])
    except (KeyError, ValueError, RuntimeError):
        return 0.0


def classificar_video(caminho_video, limiar, proporcao_suspeita, mostrar=False):
    caminho_modelo = localizar_modelo()
    modelo_deteccao = YOLO(caminho_modelo)
    modelo_pose = YOLO("yolov8n-pose.pt")
    simulador = criar_motor_fuzzy()

    captura = cv2.VideoCapture(caminho_video)
    if not captura.isOpened():
        raise FileNotFoundError(f"Não foi possível abrir o vídeo: {caminho_video}")

    fps = captura.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(captura.get(cv2.CAP_PROP_FRAME_COUNT))
    largura = int(captura.get(cv2.CAP_PROP_FRAME_WIDTH))
    altura = int(captura.get(cv2.CAP_PROP_FRAME_HEIGHT))
    escala_x = IMG_SIZE / max(largura, 1)
    escala_y = IMG_SIZE / max(altura, 1)

    riscos = []
    posicoes_anteriores = {}
    distancias_anteriores = {}
    centros_anteriores = {}
    proximo_id = 1
    frames_sem_deteccao = 0
    contador = 0

    while True:
        sucesso, frame = captura.read()
        if not sucesso:
            break

        deteccao = modelo_deteccao(frame, imgsz=IMG_SIZE, device=0, verbose=False)[0]
        pose = modelo_pose(frame, imgsz=IMG_SIZE, device=0, verbose=False)[0]
        pessoas, mochilas = obter_deteccoes(deteccao)
        esqueletos = obter_esqueletos(pose)
        centros_atuais, proximo_id = associar_ids(pessoas, centros_anteriores, proximo_id)
        centros_anteriores = centros_atuais

        distancia = 999.0
        velocidade = 0.0
        interacao_valida = False
        punho_escolhido = None
        centro_mochila_visual = None
        melhor_interacao = None
        for pessoa in pessoas:
            pessoa_id = next(id_ for id_, centro in centros_atuais.items() if np.allclose(centro, centro_caixa(pessoa)))
            pessoa_centro = centro_caixa(pessoa)
            esqueleto = min(
                esqueletos,
                key=lambda item: distancia_esqueleto_pessoa(item, pessoa_centro),
                default=None,
            )
            punhos = obter_punhos(esqueleto) if esqueleto is not None else []
            pessoa_box = pessoa.xyxy[0].cpu().numpy()
            altura_pessoa = max(float(pessoa_box[3] - pessoa_box[1]) * escala_y, 1.0)
            for mochila in mochilas:
                centro_mochila = centro_caixa(mochila)
                punho = escolher_punho(punhos, centro_mochila)
                if punho is None:
                    continue
                punho_ajustado = np.array([punho[0] * escala_x, punho[1] * escala_y])
                mochila_ajustada = np.array([centro_mochila[0] * escala_x, centro_mochila[1] * escala_y])
                distancia_atual = float(np.linalg.norm(punho_ajustado - mochila_ajustada))
                distancia_normalizada = distancia_atual / altura_pessoa
                anterior = posicoes_anteriores.get(pessoa_id)
                velocidade_atual = float(np.linalg.norm(punho_ajustado - anterior) * fps) if anterior is not None else 0.0
                distancia_anterior = distancias_anteriores.get(pessoa_id)
                movimento_em_direcao = distancia_anterior is not None and distancia_atual < distancia_anterior - 2.0
                valida = True
                candidato = (valida, distancia_atual, velocidade_atual, punho, centro_mochila, pessoa_id)
                if melhor_interacao is None or (candidato[0], -candidato[1]) > (melhor_interacao[0], -melhor_interacao[1]):
                    melhor_interacao = candidato

        if melhor_interacao is not None:
            interacao_valida, distancia, velocidade, punho_escolhido, centro_mochila_visual, pessoa_id = melhor_interacao
            if interacao_valida:
                posicoes_anteriores[pessoa_id] = np.array([punho_escolhido[0] * escala_x, punho_escolhido[1] * escala_y])
                distancias_anteriores[pessoa_id] = distancia
        else:
            frames_sem_deteccao += 1

        # Mantém o comportamento anterior: pessoa, punho e bolsa detectados
        # já habilitam a avaliação fuzzy do frame.
        risco = calcular_risco(simulador, distancia, velocidade) if interacao_valida else 0.0
        riscos.append(risco)
        contador += 1

        if mostrar:
            frame = pose.plot(
                img=frame,
                boxes=False,
                labels=False,
                conf=False,
                kpt_radius=2,
                kpt_line=True,
            )
            for mochila_id, mochila in enumerate(mochilas, start=1):
                x1, y1, x2, y2 = mochila.xyxy[0].cpu().numpy().astype(int)
                confianca = float(mochila.conf.item())
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 165, 0), 2)
                cv2.putText(
                    frame,
                    f"{mochila_id} - Bolsa - {confianca * 100:.0f}%",
                    (x1, max(y1 - 8, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 165, 0),
                    2,
                )
            for pessoa_id, pessoa in zip(centros_atuais, pessoas):
                x1, y1, x2, y2 = pessoa.xyxy[0].cpu().numpy().astype(int)
                confianca = float(pessoa.conf.item())
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"{pessoa_id} - Pessoa - {confianca * 100:.0f}%",
                    (x1, max(y1 - 8, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )
            if punho_escolhido is not None and centro_mochila_visual is not None:
                punho_xy = tuple(np.round(punho_escolhido).astype(int))
                mochila_xy = tuple(np.round(centro_mochila_visual).astype(int))
                cv2.line(frame, punho_xy, mochila_xy, (255, 255, 0), 2)
                cv2.circle(frame, punho_xy, 6, (255, 0, 255), -1)
            estado_interacao = "SUSPEITA" if interacao_valida else "sem suspeita"
            texto = f"Risco: {risco:.1f}% | Frame: {contador}/{total_frames}"
            cv2.putText(frame, texto, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(frame, estado_interacao, (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow("Classificacao fuzzy", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        if contador % 100 == 0:
            print(f"Frames processados: {contador}/{total_frames}")

    captura.release()
    if mostrar:
        cv2.destroyAllWindows()

    if not riscos:
        raise RuntimeError("Nenhum frame foi processado.")

    riscos_array = np.asarray(riscos, dtype=np.float32)
    frames_suspeitos = int(np.count_nonzero(riscos_array >= limiar))
    proporcao = frames_suspeitos / len(riscos_array)
    classificacao = "FURTO" if proporcao >= proporcao_suspeita else "NAO_FURTO"

    print("\n=== RESULTADO DO VÍDEO ===")
    print(f"Vídeo: {caminho_video}")
    print(f"Frames processados: {len(riscos_array)}")
    print(f"Risco médio: {riscos_array.mean():.2f}%")
    print(f"Risco máximo: {riscos_array.max():.2f}%")
    print(f"Frames acima do limiar: {frames_suspeitos} ({proporcao * 100:.2f}%)")
    print(f"Frames sem detecção completa: {frames_sem_deteccao}")
    print(f"Classificação: {classificacao}")

    return classificacao, riscos_array


def main():
    parser = argparse.ArgumentParser(description="Classifica um vídeo com YOLO e regras fuzzy.")
    parser.add_argument("video", help="Caminho do vídeo a classificar")
    parser.add_argument("--limiar", type=float, default=LIMIAR_RISCO_PADRAO)
    parser.add_argument("--proporcao", type=float, default=PROPORCAO_SUSPEITA_PADRAO)
    parser.add_argument("--mostrar", action="store_true", help="Exibe o vídeo durante o processamento")
    args = parser.parse_args()

    if not 0 < args.proporcao <= 1:
        parser.error("--proporcao deve estar entre 0 e 1.")
    classificar_video(args.video, args.limiar, args.proporcao, args.mostrar)


if __name__ == "__main__":
    main()
