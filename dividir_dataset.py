import os
import shutil
import random
from pathlib import Path

# Fixa a semente aleatória para garantir reprodutibilidade
random.seed(42)

def criar_estrutura_diretarios(base_saida):
    """Cria a árvore de pastas padrão do YOLO."""
    subpastas = ['images/train', 'images/val', 'images/test',
                 'labels/train', 'labels/val', 'labels/test']
    for sub in subpastas:
        Path(os.path.join(base_saida, sub)).mkdir(parents=True, exist_ok=True)

def processar_categoria(categoria, base_dir, base_saida, p_train=0.70, p_val=0.20):
    """
    Associa imagens e rótulos da categoria, realiza o shuffle e
    copia proporcionalmente para train (70%), val (20%) e test (10%).
    """
    pasta_img = os.path.join(base_dir, "02_frames_extraidos", categoria)
    pasta_txt = os.path.join(base_dir, "03_rotulos_yolo", categoria)
    
    imagens = [f for f in os.listdir(pasta_img) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    random.shuffle(imagens)
    
    total = len(imagens)
    n_train = int(total * p_train)
    n_val = int(total * p_val)
    
    train_imgs = imagens[:n_train]
    val_imgs = imagens[n_train:n_train + n_val]
    test_imgs = imagens[n_train + n_val:]
    
    divisões = [('train', train_imgs), ('val', val_imgs), ('test', test_imgs)]
    
    print(f"=== Processando Categoria: {categoria} ({total} amostras) ===")
    for split_nome, conjunto in divisões:
        cnt = 0
        for img_nome in conjunto:
            stem = Path(img_nome).stem
            txt_nome = f"{stem}.txt"
            
            src_img = os.path.join(pasta_img, img_nome)
            src_txt = os.path.join(pasta_txt, txt_nome)
            
            dst_img = os.path.join(base_saida, 'images', split_nome, img_nome)
            dst_txt = os.path.join(base_saida, 'labels', split_nome, txt_nome)
            
            shutil.copy(src_img, dst_img)
            
            # Se o arquivo .txt existir, copia; caso contrário, cria um arquivo vazio
            if os.path.exists(src_txt):
                shutil.copy(src_txt, dst_txt)
            else:
                with open(dst_txt, 'w', encoding='utf-8') as f:
                    pass
            cnt += 1
        print(f"  -> {split_nome.upper()}: {cnt} arquivos salvos.")
    print()

def criar_arquivo_yaml(base_saida):
    """Gera o arquivo data.yaml de configuração do YOLO."""
    caminho_abs = os.path.abspath(base_saida).replace("\\", "/")
    
    conteudo_yaml = f"""path: {caminho_abs}
train: images/train
val: images/val
test: images/test

nc: 2
names: ['person', 'backpack_bag']
"""
    caminho_yaml = os.path.join(base_saida, "data.yaml")
    with open(caminho_yaml, "w", encoding="utf-8") as f:
        f.write(conteudo_yaml)
    print(f"Arquivo 'data.yaml' gerado com sucesso em: {caminho_yaml}")

if __name__ == "__main__":
    base_dir = "."
    base_saida = "./dataset_yolo_final"
    
    criar_estrutura_diretarios(base_saida)
    
    processar_categoria("normais", base_dir, base_saida)
    processar_categoria("suspeitos", base_dir, base_saida)
    
    criar_arquivo_yaml(base_saida)
    print("\nDivisão concluída e Dataset YOLO Final estruturado com sucesso!")