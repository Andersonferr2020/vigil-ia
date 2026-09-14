import os
import time
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# 1. Carrega o modelo especialista em estimativa de pose
model_pose = YOLO('yolov8n-pose.pt')

def extrair_metricas_categoria(pasta_imagens, pasta_saida, classe_rotulo):
    """
    Extrai coordenadas das mãos (keypoints 9 e 10 COCO), calcula a velocidade
    e a distância aproximada, salvando em arquivo .npy.
    """
    Path(pasta_saida).mkdir(parents=True, exist_ok=True)
    imagens = sorted([f for f in os.listdir(pasta_imagens) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    
    print(f"=== Extraindo Séries Temporais Fuzzy: {pasta_imagens} ({len(imagens)} imagens) ===")
    
    amostras_extraidas = []
    posicao_anterior = None
    
    # Simulação da posição central do recipiente de referência (mochila/bolsa)
    # Em um quadro 640x640, consideramos o quadrante do corpo/recipiente
    centro_referencia = np.array([450.0, 450.0]) 
    
    contador = 0
    for img_nome in imagens:
        caminho_img = os.path.join(pasta_imagens, img_nome)
        
        # Executa a estimativa de pose na GPU
        results = model_pose(caminho_img, verbose=False, device=0)
        
        distancia = 999.0  # Valor alto padrão caso não detecte mão
        velocidade = 0.0
        
        for result in results:
            if result.keypoints is not None and len(result.keypoints) > 0:
                keypoints = result.keypoints.xy.cpu().numpy()
                confiancas = result.keypoints.conf.cpu().numpy()
                
                if len(keypoints) > 0:
                    esqueleto = keypoints[0] # Primeiro indivíduo da cena
                    confiancas_esqueleto = confiancas[0]
                    
                    # Keypoint 9 = Punho Esquerdo, Keypoint 10 = Punho Direito
                    punho_esq = esqueleto[9] if len(esqueleto) > 9 else None
                    punho_dir = esqueleto[10] if len(esqueleto) > 10 else None
                    
                    # Seleciona a mão ativa que foi detectada
                    mao_ativa = None
                    if punho_esq is not None and confiancas_esqueleto[9] > 0:
                        mao_ativa = punho_esq
                    elif punho_dir is not None and confiancas_esqueleto[10] > 0:
                        mao_ativa = punho_dir
                        
                    if mao_ativa is not None:
                        pos_atual = np.array(mao_ativa)
                        
                        # 1. Distância Euclidiana até a mochila
                        distancia = float(np.linalg.norm(pos_atual - centro_referencia))
                        
                        # 2. Velocidade do movimento da mão
                        if posicao_anterior is not None:
                            deslocamento = float(np.linalg.norm(pos_atual - posicao_anterior))
                            velocidade = deslocamento * 30.0  # Normalizado para 30 FPS
                            
                        posicao_anterior = pos_atual
        
        # Estrutura do vetor de entrada para o Fuzzy: [distancia, velocidade, classe_rotulo]
        # classe_rotulo: 0 = Normal, 1 = Suspeito
        amostras_extraidas.append([distancia, velocidade, classe_rotulo])
        
        contador += 1
        if contador % 2000 == 0:
            print(f"  -> {contador}/{len(imagens)} frames processados...")
            
    # Salva o array NumPy em arquivo .npy
    matriz_final = np.array(amostras_extraidas, dtype=np.float32)
    caminho_arquivo_npy = os.path.join(pasta_saida, f"series_fuzzy_{'suspeito' if classe_rotulo==1 else 'normal'}.npy")
    np.save(caminho_arquivo_npy, matriz_final)
    
    print(f"Concluído! {len(matriz_final)} vetores salvos em '{caminho_arquivo_npy}'.\n")

if __name__ == "__main__":
    base_dir = "."
    pasta_saida = os.path.join(base_dir, "04_dataset_fuzzy")
    
    # 1. Processa frames normais (Classe 0)
    extrair_metricas_categoria(
        pasta_imagens=os.path.join(base_dir, "02_frames_extraidos", "normais"),
        pasta_saida=pasta_saida,
        classe_rotulo=0
    )
    
    # 2. Processa frames suspeitos (Classe 1)
    extrair_metricas_categoria(
        pasta_imagens=os.path.join(base_dir, "02_frames_extraidos", "suspeitos"),
        pasta_saida=pasta_saida,
        classe_rotulo=1
    )