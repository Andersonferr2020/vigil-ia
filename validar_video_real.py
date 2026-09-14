import os
import cv2
import numpy as np
from ultralytics import YOLO
import skfuzzy as fuzzy
from skfuzzy import control as ctrl

def construir_motor_fuzzy():
    # Universo de Discurso
    distancia = ctrl.Antecedent(np.arange(0, 1001, 1), 'distancia')
    velocidade = ctrl.Antecedent(np.arange(0, 1001, 1), 'velocidade')
    risco = ctrl.Consequent(np.arange(0, 101, 1), 'risco')

    # Funções de Pertinência
    distancia['perto'] = fuzzy.trapmf(distancia.universe, [0, 0, 100, 200])
    distancia['media'] = fuzzy.trimf(distancia.universe, [100, 250, 400])
    distancia['longe'] = fuzzy.trapmf(distancia.universe, [450, 600, 1000, 1000])

    velocidade['lenta'] = fuzzy.trapmf(velocidade.universe, [0, 0, 100, 250])
    velocidade['moderada'] = fuzzy.trimf(velocidade.universe, [100, 300, 500])
    velocidade['rapida'] = fuzzy.trapmf(velocidade.universe, [500, 700, 1000, 1000])

    risco['baixo'] = fuzzy.trapmf(risco.universe, [0, 0, 25, 50])
    risco['medio'] = fuzzy.trimf(risco.universe, [25, 50, 75])
    risco['alto'] = fuzzy.trapmf(risco.universe, [50, 75, 100, 100])

    # Regras Mamdani
    regras = [
        ctrl.Rule(distancia['perto'] & velocidade['rapida'], risco['alto']),
        ctrl.Rule(distancia['perto'] & velocidade['moderada'], risco['alto']),
        ctrl.Rule(distancia['perto'] & velocidade['lenta'], risco['medio']),
        ctrl.Rule(distancia['media'] & velocidade['rapida'], risco['alto']),
        ctrl.Rule(distancia['media'] & velocidade['moderada'], risco['medio']),
        ctrl.Rule(distancia['media'] & velocidade['lenta'], risco['baixo']),
        ctrl.Rule(distancia['longe'], risco['baixo'])
    ]

    sistema = ctrl.ControlSystem(regras)
    return ctrl.ControlSystemSimulation(sistema)

def processar_video(caminho_video_entrada, caminho_video_saida):
    print("=== INICIANDO VALIDAÇÃO DO VIGIL-IA EM VÍDEO REAL ===")
    
    # 1. Carrega o modelo YOLO treinado
    caminhos_best = [
        os.path.abspath('runs/detect/treino_vigilia/weights/best.pt'),
        os.path.abspath('runs/detect/runs/detect/treino_vigilia/weights/best.pt'),
    ]
    caminho_weights = next((c for c in caminhos_best if os.path.exists(c)), None)
    if caminho_weights is None:
        print("ERRO: Pesos não encontrados nos diretórios de treinamento.")
        return

    model = YOLO(caminho_weights)
    simulador_fuzzy = construir_motor_fuzzy()

    # 2. Abre o vídeo de entrada
    cap = cv2.VideoCapture(caminho_video_entrada)
    if not cap.isOpened():
        print(f"ERRO: Não foi possível abrir o vídeo: {caminho_video_entrada}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps == 0: fps = 30

    # Configura o gravador de vídeo de saída
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(caminho_video_saida, fourcc, fps, (width, height))

    pos_anterior_pessoa = None
    frame_count = 0

    print(f"Processando vídeo: {caminho_video_entrada} ({width}x{height} @ {fps} FPS)...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        
        # Inferência do YOLO no quadro atual
        results = model.predict(frame, imgsz=640, conf=0.35, verbose=False)

        pessoas = []
        bolsas = []

        # Separa as detecções de pessoas e bolsas
        for box in results[0].boxes:
            cls_id = int(box.cls)
            conf = float(box.conf)
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            
            # Cálculo do centroide (cx, cy)
            cx = int((xyxy[0] + xyxy[2]) / 2)
            cy = int((xyxy[1] + xyxy[3]) / 2)

            if cls_id == 0:  # Classe person
                pessoas.append((xyxy, (cx, cy), conf))
            elif cls_id == 1:  # Classe backpack_bag
                bolsas.append((xyxy, (cx, cy), conf))

        risco_calculado = 0.0
        cor_alerta = (0, 255, 0) # Verde por padrão (Normal)
        texto_alerta = "STATUS: NORMAL"

        # Se houver pelo menos uma pessoa e uma bolsa na cena
        if len(pessoas) > 0 and len(bolsas) > 0:
            p_box, (px, py), p_conf = pessoas[0]
            b_box, (bx, by), b_conf = bolsas[0]

            # 1. Distância Euclidiana entre os centroides (px)
            dist_px = float(np.sqrt((px - bx)**2 + (py - by)**2))

            # 2. Velocidade estimada entre quadros (px/frame)
            if pos_anterior_pessoa is not None:
                vel_px = float(np.sqrt((px - pos_anterior_pessoa[0])**2 + (py - pos_anterior_pessoa[1])**2))
            else:
                vel_px = 0.0
            pos_anterior_pessoa = (px, py)

            # 3. Avaliação Fuzzy Mamdani
            d_clip = np.clip(dist_px, 0, 1000)
            v_clip = np.clip(vel_px, 0, 1000)

            simulador_fuzzy.input['distancia'] = d_clip
            simulador_fuzzy.input['velocidade'] = v_clip
            try:
                simulador_fuzzy.compute()
                risco_calculado = simulador_fuzzy.output['risco']
            except:
                risco_calculado = 0.0

            # Desenha linha entre a pessoa e a bolsa
            cv2.line(frame, (px, py), (bx, by), (255, 255, 0), 2)

            # Define estado de alerta com base no limiar ótimo de 42.5%
            if risco_calculado >= 42.5:
                cor_alerta = (0, 0, 255) # Vermelho
                texto_alerta = f"ALERTA: SUSPEITA DE FURTO ({risco_calculado:.1f}%)"
            else:
                cor_alerta = (0, 255, 0) # Verde
                texto_alerta = f"STATUS: NORMAL ({risco_calculado:.1f}%)"
        else:
            pos_anterior_pessoa = None

        # Desenha Bounding Boxes
        for box, (cx, cy), conf in pessoas:
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), cor_alerta, 2)
            cv2.putText(frame, f"Pessoa {conf:.2f}", (box[0], max(box[1] - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, cor_alerta, 2)

        for box, (cx, cy), conf in bolsas:
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (255, 165, 0), 2)
            cv2.putText(frame, f"Bolsa {conf:.2f}", (box[0], max(box[1] - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)

        # Barra Superior de Alerta
        cv2.rectangle(frame, (0, 0), (width, 50), (0, 0, 0), -1)
        cv2.putText(frame, texto_alerta, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, cor_alerta, 2)

        out.write(frame)

    cap.release()
    out.release()
    print(f"\nProcessamento concluído! Vídeo resultante salvo em: {caminho_video_saida}")

    # Reproduzir o vídeo resultante
    reproduzir_video(caminho_video_saida)

def reproduzir_video(caminho_video, largura_janela=1024, altura_janela=768):
    print(f"\nReproduzindo vídeo de resultado: {caminho_video} (Pressione 'q' ou 'ESC' para fechar)...")
    cap = cv2.VideoCapture(caminho_video)
    if not cap.isOpened():
        print(f"ERRO: Não foi possível abrir o vídeo para reprodução: {caminho_video}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    delay = int(1000 / fps) if fps > 0 else 33

    nome_janela = "Vigil-IA - Resultado"
    cv2.namedWindow(nome_janela, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(nome_janela, largura_janela, altura_janela)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow(nome_janela, frame)
        tecla = cv2.waitKey(delay) & 0xFF
        if tecla == ord('q') or tecla == 27:  # 'q' ou ESC
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Substitua pelo caminho de um vídeo de teste que você tenha no seu computador
    video_input = ".\\01_videos_brutos\\suspeitos\\Shoplifting (2).mp4" 
    video_output = "resultado_vigilia.mp4"
    
    if os.path.exists(video_input):
        processar_video(video_input, video_output)
    else:
        print(f"Coloque um vídeo de teste com o nome '{video_input}' na pasta do projeto e execute novamente.")