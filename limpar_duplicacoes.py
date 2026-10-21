

from pathlib import Path
from collections import defaultdict
import csv

BASE = Path("labels_revisados")
RELATORIO = Path("auditoria_rotulos/possiveis_duplicacoes.csv")
SAIDA = Path("auditoria_rotulos/simulacao_limpeza.csv")

LIMIAR_IOU = 0.95
CLASSE_ALVO = "1"  # backpack_bag

# MODO SIMULACAO: nenhum rotulo sera modificado.

pares_por_arquivo = defaultdict(list)

with RELATORIO.open(
    "r", encoding="utf-8-sig", newline=""
) as arquivo:
    leitor = csv.DictReader(arquivo)

    for registro in leitor:
        if registro["split"] != "train":
            continue

        iou = float(registro["iou"])

        if iou < LIMIAR_IOU:
            continue

        nome = registro["arquivo"]
        linha_a = int(registro["linha_1"])
        linha_b = int(registro["linha_2"])

        pares_por_arquivo[nome].append(
            (linha_a, linha_b, iou)
        )

resultados = []
total_pares = 0
total_linhas = 0
arquivos_validos = 0

for nome, pares in sorted(pares_por_arquivo.items()):
    caminho = BASE / "train" / nome

    if not caminho.exists():
        print("ATENCAO: arquivo nao encontrado:", caminho)
        continue

    linhas = caminho.read_text(
        encoding="utf-8"
    ).splitlines()

    grafo = defaultdict(set)

    for a, b, iou in pares:
        if not (1 <= a <= len(linhas) and 1 <= b <= len(linhas)):
            print("ATENCAO: linha invalida em", nome)
            continue

        # Verifica se ambas as caixas sao backpack_bag.
        dados_a = linhas[a - 1].split()
        dados_b = linhas[b - 1].split()

        if not dados_a or not dados_b:
            print("IGNORADO: anotacao vazia em", nome)
            continue

        if dados_a[0] != CLASSE_ALVO or dados_b[0] != CLASSE_ALVO:
            print("IGNORADO: par de outra classe em", nome)
            continue

        grafo[a].add(b)
        grafo[b].add(a)
        total_pares += 1

    if not grafo:
        continue

    arquivos_validos += 1

    visitados = set()
    remover = set()

    for inicio in sorted(grafo):
        if inicio in visitados:
            continue

        pilha = [inicio]
        componente = set()

        while pilha:
            atual = pilha.pop()

            if atual in visitados:
                continue

            visitados.add(atual)
            componente.add(atual)
            pilha.extend(grafo[atual] - visitados)

        # Mantem a primeira linha do grupo.
        manter = min(componente)

        for numero in sorted(componente):
            if numero != manter:
                remover.add(numero)

                resultados.append({
                    "arquivo": nome,
                    "linha_mantida": manter,
                    "linha_candidata_remocao": numero,
                    "conteudo": linhas[numero - 1]
                })

    total_linhas += len(remover)

SAIDA.parent.mkdir(parents=True, exist_ok=True)

with SAIDA.open(
    "w", encoding="utf-8-sig", newline=""
) as arquivo:
    campos = [
        "arquivo",
        "linha_mantida",
        "linha_candidata_remocao",
        "conteudo"
    ]

    escritor = csv.DictWriter(
        arquivo, fieldnames=campos
    )
    escritor.writeheader()
    escritor.writerows(resultados)

print("\n=== SIMULACAO DE LIMPEZA ===")
print("Conjunto analisado: train")
print("Classe analisada: backpack_bag (1)")
print("Limiar IoU:", LIMIAR_IOU)
print("Pares validos analisados:", total_pares)
print("Arquivos com pares validos:", arquivos_validos)
print("Linhas candidatas a remocao:", total_linhas)
print("Relatorio:", SAIDA.resolve())
print("\nNENHUM ROTULO ORIGINAL OU REVISADO FOI MODIFICADO.")
