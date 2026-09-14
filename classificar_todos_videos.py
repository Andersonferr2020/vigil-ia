import argparse
import random
from pathlib import Path

import numpy as np

from classificar_video import (
    LIMIAR_RISCO_PADRAO,
    PROPORCAO_SUSPEITA_PADRAO,
    classificar_video,
)

EXTENSOES_VIDEO = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}


def selecionar_videos(pasta, quantidade, semente):
    gerador = random.Random(semente)
    videos_selecionados = []
    raiz = Path(pasta)

    for subpasta_nome in ("normais", "suspeitos"):
        subpasta = raiz / subpasta_nome
        videos = sorted(
            caminho
            for caminho in subpasta.rglob("*")
            if caminho.is_file() and caminho.suffix.lower() in EXTENSOES_VIDEO
        )
        quantidade_selecionada = min(quantidade, len(videos))
        selecionados = gerador.sample(videos, quantidade_selecionada)
        videos_selecionados.extend(sorted(selecionados))
        print(
            f"{subpasta_nome}: {len(videos)} disponíveis, "
            f"{quantidade_selecionada} selecionados aleatoriamente."
        )

    return videos_selecionados


def classe_esperada(caminho):
    partes = {parte.lower() for parte in caminho.parts}
    if "suspeitos" in partes:
        return "FURTO"
    if "normais" in partes:
        return "NAO_FURTO"
    return "DESCONHECIDA"


def processar_videos(pasta, quantidade, semente, limiar, proporcao_suspeita, mostrar):
    videos = selecionar_videos(pasta, quantidade, semente)
    if not videos:
        raise FileNotFoundError(f"Nenhum vídeo encontrado em: {pasta}")

    resultados = []
    print(f"Encontrados {len(videos)} vídeos em '{pasta}'.")

    for indice, video in enumerate(videos, start=1):
        print(f"\n[{indice}/{len(videos)}] Processando: {video}")
        try:
            classificacao, riscos = classificar_video(
                str(video),
                limiar=limiar,
                proporcao_suspeita=proporcao_suspeita,
                mostrar=mostrar,
            )
            riscos = np.asarray(riscos)
            percentual = float(np.mean(riscos >= limiar) * 100)
            resultados.append(
                {
                    "video": str(video),
                    "esperado": classe_esperada(video),
                    "classificacao": classificacao,
                    "risco_medio": float(riscos.mean()),
                    "risco_maximo": float(riscos.max()),
                    "frames_suspeitos": percentual,
                    "status": "OK",
                }
            )
        except Exception as erro:
            resultados.append(
                {
                    "video": str(video),
                    "esperado": classe_esperada(video),
                    "classificacao": "ERRO",
                    "risco_medio": 0.0,
                    "risco_maximo": 0.0,
                    "frames_suspeitos": 0.0,
                    "status": str(erro),
                }
            )
            print(f"Erro neste vídeo: {erro}")

    return resultados


def imprimir_tabela(resultados):
    colunas = [
        ("#", 3),
        ("Vídeo", 38),
        ("Esperado", 13),
        ("Resultado", 13),
        ("Risco médio", 12),
        ("Risco máx.", 11),
        ("Frames > limiar", 16),
        ("Status", 8),
    ]
    cabecalho = " | ".join(nome.ljust(largura) for nome, largura in colunas)
    separador = "-+-".join("-" * largura for _, largura in colunas)

    print("\n" + "=" * len(cabecalho))
    print("TABELA FINAL DOS VÍDEOS")
    print("=" * len(cabecalho))
    print(cabecalho)
    print(separador)

    for indice, resultado in enumerate(resultados, start=1):
        nome_video = Path(resultado["video"]).name
        if len(nome_video) > 38:
            nome_video = nome_video[:35] + "..."
        valores = [
            str(indice),
            nome_video,
            resultado["esperado"],
            resultado["classificacao"],
            f"{resultado['risco_medio']:.2f}%",
            f"{resultado['risco_maximo']:.2f}%",
            f"{resultado['frames_suspeitos']:.2f}%",
            resultado["status"] if resultado["status"] != "OK" else "OK",
        ]
        print(" | ".join(valor.ljust(largura)[:largura] for valor, (_, largura) in zip(valores, colunas)))


def main():
    parser = argparse.ArgumentParser(
        description="Classifica uma amostra aleatória de vídeos usando YOLO e fuzzy."
    )
    parser.add_argument(
        "--pasta",
        default="01_videos_brutos",
        help="Pasta raiz dos vídeos (padrão: 01_videos_brutos)",
    )
    parser.add_argument("--limiar", type=float, default=LIMIAR_RISCO_PADRAO)
    parser.add_argument("--proporcao", type=float, default=PROPORCAO_SUSPEITA_PADRAO)
    parser.add_argument(
        "--quantidade",
        type=int,
        default=10,
        help="Quantidade de vídeos selecionados por subpasta (padrão: 10)",
    )
    parser.add_argument(
        "--semente",
        type=int,
        default=42,
        help="Semente da seleção aleatória (padrão: 42)",
    )
    parser.add_argument(
        "--mostrar",
        action="store_true",
        help="Exibe cada vídeo durante o processamento; pressione q para avançar.",
    )
    args = parser.parse_args()

    if not 0 < args.proporcao <= 1:
        parser.error("--proporcao deve estar entre 0 e 1.")
    if args.quantidade < 1:
        parser.error("--quantidade deve ser maior que zero.")

    resultados = processar_videos(
        args.pasta,
        args.quantidade,
        args.semente,
        args.limiar,
        args.proporcao,
        mostrar=args.mostrar,
    )
    imprimir_tabela(resultados)


if __name__ == "__main__":
    main()
