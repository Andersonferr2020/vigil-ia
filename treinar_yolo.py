import os
from ultralytics import YOLO

def executar_treinamento_leve():
    print("=== INICIANDO TREINAMENTO YOLOv8n ===")

    caminhos_checkpoint = [
        os.path.abspath('./runs/detect/treino_vigilia/weights/last.pt'),
    ]
    caminho_checkpoint = next(
        (caminho for caminho in caminhos_checkpoint if os.path.exists(caminho)),
        None
    )
    retomando = caminho_checkpoint is not None
    model = YOLO(caminho_checkpoint if retomando else 'yolov8n.pt')
    caminho_yaml = os.path.abspath('./dataset_yolo_final/data.yaml')
    pasta_projeto = 'runs/detect'
    
    results = model.train(
        data=caminho_yaml,
        epochs=50,
        patience=10,        # Early stopping após 10 épocas sem melhora
        imgsz=640,          # Preserva mais detalhes dos objetos
        batch=8,            # Configuração solicitada; pode exigir mais RAM/VRAM
        device=0,           # GPU CUDA
        workers=2,          # Evita picos de RAM no Windows
        cache=False,        # Não carrega as 20 mil imagens descompactadas na RAM
        mosaic=1.0,         # Ativa a augmentação Mosaic
        close_mosaic=10,    # Desativa Mosaic nas últimas 10 épocas para estabilizar
        project=pasta_projeto,
        name='treino_vigilia',
        exist_ok=True,      # Reutiliza a mesma pasta, sem criar pastas numeradas
        verbose=False,      # OTIMIZAÇÃO: Desativa logs contínuos por lote no terminal
        plots=True,         # Mantém a geração dos gráficos das curvas ao final
        resume=retomando    # Continua do último checkpoint quando disponível
    )
    
    print("\n=== TREINAMENTO CONCLUÍDO COM SUCESSO! ===")
    print("Resultados salvos em: runs/detect/treino_vigilia")

if __name__ == "__main__":
    executar_treinamento_leve()