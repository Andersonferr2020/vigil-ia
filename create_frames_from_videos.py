import os
import cv2
from pathlib import Path

def extrair_frames_categoria(pasta_entrada, pasta_saida, intervalo_frames=3):
    """
    Varre os vídeos de 'pasta_entrada' e salva os quadros a cada N frames em 'pasta_saida'.
    """
    Path(pasta_saida).mkdir(parents=True, exist_ok=True)
    
    extensoes_suportadas = ('.mp4', '.avi', '.mov', '.mkv')
    videos = [f for f in os.listdir(pasta_entrada) if f.lower().endswith(extensoes_suportadas)]
    
    print(f"=== Processando diretório: {pasta_entrada} ===")
    print(f"Total de vídeos encontrados: {len(videos)}")
    
    total_frames = 0
    for video_nome in videos:
        caminho_video = os.path.join(pasta_entrada, video_nome)
        cap = cv2.VideoCapture(caminho_video)
        
        nome_base = os.path.splitext(video_nome)
        count = 0
        frame_id = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Amostragem: salva 1 quadro a cada 'intervalo_frames'
            if count % intervalo_frames == 0:
                nome_frame = f"{nome_base}_f{frame_id:05d}.jpg"
                caminho_salvar = os.path.join(pasta_saida, nome_frame)
                cv2.imwrite(caminho_salvar, frame)
                frame_id += 1
                total_frames += 1
                
            count += 1
            
        cap.release()
        print(f"  -> Vídeo '{video_nome}': {frame_id} imagens geradas.")
        
    print(f"Concluído! Total de imagens salvas em '{pasta_saida}': {total_frames}\n")

# Execução do fatiamento local
if __name__ == "__main__":
    base_dir = "."  # Executando dentro de 'vigilia_dataset'
    
    # 1. Fatiar vídeos de ações normais
    extrair_frames_categoria(
        pasta_entrada=os.path.join(base_dir, "01_videos_brutos", "normais"),
        pasta_saida=os.path.join(base_dir, "02_frames_extraidos", "normais"),
        intervalo_frames=3  # Extrai 1 a cada 3 quadros
    )
    
    # 2. Fatiar vídeos de ações suspeitas
    extrair_frames_categoria(
        pasta_entrada=os.path.join(base_dir, "01_videos_brutos", "suspeitos"),
        pasta_saida=os.path.join(base_dir, "02_frames_extraidos", "suspeitos"),
        intervalo_frames=3
    )