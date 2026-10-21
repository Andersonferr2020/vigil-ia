
from pathlib import Path
import cv2

BASE = Path("dataset_yolo_final")
PASTA_IMAGENS = BASE / "images" / "test"
PASTA_LABELS = BASE / "labels" / "test"
PASTA_SAIDA = Path("inspecao_rotulos")

PASTA_SAIDA.mkdir(exist_ok=True)

CLASSES = {
    0: ("person", (0, 255, 0)),
    1: ("backpack_bag", (0, 0, 255)),
}

# Seleciona até 10 imagens com anotações de bolsas/mochilas.
selecionados = []

for arquivo_txt in sorted(PASTA_LABELS.glob("*.txt")):
    linhas = arquivo_txt.read_text(encoding="utf-8").splitlines()

    if any(linha.strip().startswith("1 ") for linha in linhas):
        selecionados.append(arquivo_txt)

    if len(selecionados) >= 10:
        break

for arquivo_txt in selecionados:
    imagem_path = PASTA_IMAGENS / (arquivo_txt.stem + ".jpg")

    if not imagem_path.exists():
        print("Imagem não encontrada:", imagem_path)
        continue

    imagem = cv2.imread(str(imagem_path))

    if imagem is None:
        print("Erro ao abrir:", imagem_path)
        continue

    altura, largura = imagem.shape[:2]

    for linha in arquivo_txt.read_text(encoding="utf-8").splitlines():
        dados = linha.split()

        if len(dados) != 5:
            continue

        classe = int(dados[0])
        xc, yc, w, h = map(float, dados[1:])

        if classe not in CLASSES:
            continue

        nome, cor = CLASSES[classe]

        x1 = int((xc - w / 2) * largura)
        y1 = int((yc - h / 2) * altura)
        x2 = int((xc + w / 2) * largura)
        y2 = int((yc + h / 2) * altura)

        cv2.rectangle(imagem, (x1, y1), (x2, y2), cor, 2)
        cv2.putText(
            imagem,
            nome,
            (max(0, x1), max(20, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            cor,
            2,
        )

    destino = PASTA_SAIDA / imagem_path.name
    cv2.imwrite(str(destino), imagem)
    print("Salvo:", destino)

print("\nInspeção concluída.")
print("Imagens salvas em:", PASTA_SAIDA.resolve())
