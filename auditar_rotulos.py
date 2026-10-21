
from pathlib import Path
from itertools import combinations
import csv

BASE = Path("dataset_yolo_final")
SAIDA = Path("auditoria_rotulos")
SAIDA.mkdir(exist_ok=True)

CLASSE_ALVO = 1  # backpack_bag
LIMIAR_IOU = 0.50

def converter_caixa(xc, yc, w, h):
    return (
        xc - w / 2,
        yc - h / 2,
        xc + w / 2,
        yc + h / 2
    )

def calcular_iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)

    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])

    uniao = area_a + area_b - inter

    return inter / uniao if uniao > 0 else 0

resultados = []
resumo = {}

for split in ["train", "val", "test"]:
    pasta = BASE / "labels" / split
    arquivos = sorted(pasta.glob("*.txt"))

    imagens_suspeitas = set()
    pares_suspeitos = 0
    caixas_classe = 0

    for arquivo in arquivos:
        caixas = []

        for numero_linha, linha in enumerate(
            arquivo.read_text(encoding="utf-8").splitlines(), start=1
        ):
            partes = linha.split()

            if len(partes) != 5:
                continue

            try:
                classe = int(partes[0])
                valores = list(map(float, partes[1:]))
            except ValueError:
                continue

            if classe != CLASSE_ALVO:
                continue

            caixas_classe += 1
            caixas.append((
                numero_linha,
                converter_caixa(*valores)
            ))

        for (linha_a, caixa_a), (linha_b, caixa_b) in combinations(caixas, 2):
            iou = calcular_iou(caixa_a, caixa_b)

            if iou >= LIMIAR_IOU:
                imagens_suspeitas.add(arquivo.name)
                pares_suspeitos += 1

                resultados.append({
                    "split": split,
                    "arquivo": arquivo.name,
                    "linha_1": linha_a,
                    "linha_2": linha_b,
                    "iou": round(iou, 4)
                })

    resumo[split] = {
        "arquivos": len(arquivos),
        "caixas": caixas_classe,
        "imagens_suspeitas": len(imagens_suspeitas),
        "pares_suspeitos": pares_suspeitos
    }

arquivo_csv = SAIDA / "possiveis_duplicacoes.csv"

with arquivo_csv.open("w", newline="", encoding="utf-8-sig") as f:
    campos = ["split", "arquivo", "linha_1", "linha_2", "iou"]
    escritor = csv.DictWriter(f, fieldnames=campos)
    escritor.writeheader()
    escritor.writerows(resultados)

print("\n=== AUDITORIA DE ROTULOS BACKPACK_BAG ===")
print(f"Limiar de IoU: {LIMIAR_IOU}")

for split, dados in resumo.items():
    print(f"\nConjunto: {split}")
    print(f"Arquivos analisados: {dados['arquivos']}")
    print(f"Caixas backpack_bag: {dados['caixas']}")
    print(f"Imagens com sobreposicoes suspeitas: {dados['imagens_suspeitas']}")
    print(f"Pares de caixas suspeitos: {dados['pares_suspeitos']}")

print("\nRelatorio salvo em:", arquivo_csv.resolve())
print("Nenhum rotulo original foi alterado.")
