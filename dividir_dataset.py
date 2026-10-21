
import random
import shutil
from pathlib import Path

SEMENTE = 42
PROPORCOES = (0.70, 0.15, 0.15)

EXTENSOES = {".jpg", ".jpeg", ".png"}


def identificar_video(nome_imagem):
    """Remove o sufixo do frame, por exemplo _f0000120."""
    stem = Path(nome_imagem).stem
    prefixo, separador, numero = stem.rpartition("_f")

    if separador and numero.isdigit():
        return prefixo

    raise ValueError(
        f"Nome de frame fora do padrão esperado: {nome_imagem}"
    )


def dividir_videos(videos, semente):
    videos = sorted(videos)
    random.Random(semente).shuffle(videos)

    total = len(videos)
    n_train = int(total * PROPORCOES[0])
    n_val = int(total * PROPORCOES[1])

    return {
        "train": set(videos[:n_train]),
        "val": set(videos[n_train:n_train + n_val]),
        "test": set(videos[n_train + n_val:]),
    }


def main():
    base_dir = Path(__file__).resolve().parent
    saida = base_dir / "dataset_yolo_final"

    categorias = {
        "normais": "normal",
        "suspeitos": "suspeito",
    }

    # Validar todas as entradas antes de copiar arquivos.
    dados = {}

    for categoria in categorias:
        pasta_img = base_dir / "02_frames_extraidos" / categoria
        pasta_txt = base_dir / "03_rotulos_yolo" / categoria

        if not pasta_img.is_dir() or not pasta_txt.is_dir():
            raise FileNotFoundError(
                f"Não encontrei as pastas de imagens/rótulos de "
                f"{categoria}."
            )

        imagens = sorted(
            p for p in pasta_img.iterdir()
            if p.is_file() and p.suffix.lower() in EXTENSOES
        )

        if not imagens:
            raise RuntimeError(f"Nenhuma imagem encontrada em {pasta_img}")

        faltantes = [
            img.name for img in imagens
            if not (pasta_txt / f"{img.stem}.txt").is_file()
        ]

        if faltantes:
            raise RuntimeError(
                f"{len(faltantes)} imagens de {categoria} não têm rótulo. "
                f"Exemplo: {faltantes[0]}"
            )

        por_video = {}
        for img in imagens:
            video = identificar_video(img.name)
            por_video.setdefault(video, []).append(img)

        dados[categoria] = {
            "pasta_img": pasta_img,
            "pasta_txt": pasta_txt,
            "por_video": por_video,
        }

    # Não misturar com uma saída anterior.
    if saida.exists() and any(saida.iterdir()):
        raise FileExistsError(
            f"A pasta {saida} já existe e não está vazia. "
            "Não vou sobrescrever seus arquivos. Renomeie ou remova "
            "essa pasta conscientemente antes de executar novamente."
        )

    saida.mkdir(parents=True, exist_ok=True)

    for subpasta in (
        "images/train", "images/val", "images/test",
        "labels/train", "labels/val", "labels/test",
    ):
        (saida / subpasta).mkdir(parents=True, exist_ok=True)

    videos_por_split = {s: set() for s in ("train", "val", "test")}
    contagens = {s: 0 for s in ("train", "val", "test")}

    for categoria, info in dados.items():
        grupos = info["por_video"]
        divisao = dividir_videos(grupos.keys(), SEMENTE)

        print(
            f"\n{categoria}: {len(grupos)} vídeos, "
            f"{sum(map(len, grupos.values()))} imagens"
        )

        for split, videos in divisao.items():
            for video in sorted(videos):
                videos_por_split[split].add((categoria, video))

                for img in grupos[video]:
                    # Prefixo impede colisões entre categorias.
                    nome = f"{categorias[categoria]}_{img.name}"
                    nome_txt = f"{Path(nome).stem}.txt"

                    shutil.copy2(
                        img, saida / "images" / split / nome
                    )
                    shutil.copy2(
                        info["pasta_txt"] / f"{img.stem}.txt",
                        saida / "labels" / split / nome_txt,
                    )
                    contagens[split] += 1

            print(f"  {split}: {len(videos)} vídeos")

    # Verificar que nenhum vídeo aparece em mais de um conjunto.
    conjuntos = [
        {video for _, video in videos_por_split[s]}
        for s in ("train", "val", "test")
    ]

    # A verificação considera também a categoria, pois nomes de vídeo
    # podem se repetir entre categorias.
    pares = [
        videos_por_split["train"] & videos_por_split["val"],
        videos_por_split["train"] & videos_por_split["test"],
        videos_por_split["val"] & videos_por_split["test"],
    ]

    if any(pares):
        raise RuntimeError("Um vídeo foi distribuído entre conjuntos.")

    yaml = (
        f"path: {saida.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n\n"
        "nc: 2\n"
        "names: ['person', 'backpack_bag']\n"
    )

    (saida / "data.yaml").write_text(yaml, encoding="utf-8")

    print("\nDivisão concluída:")
    total = sum(contagens.values())
    for split, quantidade in contagens.items():
        print(f"  {split}: {quantidade} imagens")

    print(f"  total: {total} imagens")
    print(f"Configuração: {saida / 'data.yaml'}")
    print(
        "\nAtenção: os rótulos são pseudo-rótulos gerados "
        "automaticamente, não anotações originais do TCC."
    )


if __name__ == "__main__":
    main()
