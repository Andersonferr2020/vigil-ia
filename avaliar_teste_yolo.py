import os
from ultralytics import YOLO

def avaliar_modelo_teste():
    print("=== INICIANDO TAREFA 3: AVALIAÇÃO NO CONJUNTO DE TESTE INDEPENDENTE ===")
    
    # 1. Carrega o melhor modelo treinado, aceitando o caminho legado duplicado
    caminhos_best = [
        os.path.abspath('./runs/detect/treino_vigilia/weights/best.pt'),
        os.path.abspath('./runs/detect/runs/detect/treino_vigilia/weights/best.pt'),
    ]
    caminho_best = next((caminho for caminho in caminhos_best if os.path.exists(caminho)), None)
    if caminho_best is None:
        raise FileNotFoundError('Nenhum best.pt foi encontrado nos diretórios de treinamento.')
    caminho_yaml = os.path.abspath('./dataset_yolo_final/data.yaml')
    
    model = YOLO(caminho_best)
    
    # 2. Executa a avaliação no split de Teste
    metrics = model.val(
        data=caminho_yaml,
        split='test',       # Avalia exclusivamente na pasta test/
        imgsz=640,
        batch=4,            # Adequado aos 2 GB de VRAM da MX250
        device=0,           # GPU
        workers=1,          # Carregamento estável no Windows
        verbose=True,       # Exibe o progresso e as métricas por classe
        plots=True,         # Gera matriz de confusão e curvas
        project='runs/detect',
        name='avaliacao_teste',
        exist_ok=True
    )
    
    print("\n=== MÉTRICAS FINAIS NO CONJUNTO DE TESTE ===")
    print(f"Precisão (Box P):  {metrics.box.mp:.4f}")
    print(f"Recall (Box R):    {metrics.box.mr:.4f}")
    print(f"mAP50:             {metrics.box.map50:.4f}")
    print(f"mAP50-95:          {metrics.box.map:.4f}")
    print(f"\nResultados e matriz de confusão salvos em: runs/detect/avaliacao_teste")

if __name__ == "__main__":
    avaliar_modelo_teste()