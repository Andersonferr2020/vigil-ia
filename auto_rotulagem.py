import os
from pathlib import Path
from ultralytics import YOLO

# 1. Carrega o modelo YOLO pré-treinado na base COCO
model = YOLO('yolov8n.pt')

# Mapeamento das classes COCO de interesse:
# COCO Class 0 = Person  -> Mapeado para ID 0 (pessoa)
# COCO Class 24 = Backpack -> Mapeado para ID 1 (mochila)
# COCO Class 26 = Handbag  -> Mapeado para ID 1 (bolsa - agrupados como 'backpack_bag')
COCO_CLASSES_INTERESSE = {0: 0, 24: 1, 26: 1}

def gerar_rotulos_automaticos(pasta_imagens, pasta_rotulos):
    """
    Realiza a inferência nas imagens e gera arquivos .txt no formato YOLO.
    """
    Path(pasta_rotulos).mkdir(parents=True, exist_ok=True)
    
    imagens = [f for f in os.listdir(pasta_imagens) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"=== Gerando rótulos para: {pasta_imagens} ({len(imagens)} imagens) ===")
    
    contador = 0
    for img_nome in imagens:
        caminho_img = os.path.join(pasta_imagens, img_nome)
        
        # Extrai o nome do arquivo limpo sem extensão
        nome_base = Path(img_nome).stem
        caminho_txt = os.path.join(pasta_rotulos, f"{nome_base}.txt")
        
        # Executa inferência
        results = model(caminho_img, verbose=False, device=0)
        
        linhas_txt = []
        for result in results:
            boxes = result.boxes
            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    cls_id = int(box.cls.item())
                    
                    # Filtra apenas as classes de interesse (pessoa, mochila, bolsa)
                    if cls_id in COCO_CLASSES_INTERESSE:
                        novo_cls_id = COCO_CLASSES_INTERESSE[cls_id]
                        
                        # box.xywhn tem formato (1, 4); seleciona a única caixa.
                        x, y, w, h = box.xywhn[0].tolist()
                        
                        linhas_txt.append(f"{novo_cls_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
        
        # Escreve o arquivo de anotação .txt
        with open(caminho_txt, 'w', encoding='utf-8') as f:
            f.writelines(linhas_txt)
            
        contador += 1
        if contador % 2000 == 0:
            print(f"  -> {contador}/{len(imagens)} imagens processadas...")
            
    print(f"Concluído! {contador} arquivos .txt gerados em '{pasta_rotulos}'.\n")

if __name__ == "__main__":
    base_dir = "."
    
    # 1. Rotular imagens normais
    gerar_rotulos_automaticos(
        pasta_imagens=os.path.join(base_dir, "02_frames_extraidos", "normais"),
        pasta_rotulos=os.path.join(base_dir, "03_rotulos_yolo", "normais")
    )
    
    # 2. Rotular imagens suspeitas
    gerar_rotulos_automaticos(
        pasta_imagens=os.path.join(base_dir, "02_frames_extraidos", "suspeitos"),
        pasta_rotulos=os.path.join(base_dir, "03_rotulos_yolo", "suspeitos")
    )