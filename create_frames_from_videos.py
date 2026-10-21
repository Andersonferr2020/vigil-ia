
import cv2
from pathlib import Path


def extrair_frames_categoria(pasta_entrada, pasta_saida, intervalo_frames=3):
    pasta_entrada = Path(pasta_entrada)
    pasta_saida = Path(pasta_saida)

    if not pasta_entrada.exists():
        print(f"[ERRO] Pasta não encontrada: {pasta_entrada}")
        return

    pasta_saida.mkdir(parents=True, exist_ok=True)

    extensoes = {".mp4", ".avi", ".mov", ".mkv"}
    videos = sorted(
        p for p in pasta_entrada.iterdir()
        if p.is_file() and p.suffix.lower() in extensoes
    )

    print(f"\n=== Processando: {pasta_entrada} ===")
    print(f"Vídeos encontrados: {len(videos)}")

    total_frames = 0

    for video in videos:
        cap = cv2.VideoCapture(str(video))

        if not cap.isOpened():
            print(f"[AVISO] Não foi possível abrir: {video.name}")
            continue

        nome_base = video.stem
        count = 0
        frames_salvos = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if count % intervalo_frames == 0:
                nome_frame = f"{nome_base}_f{count:07d}.jpg"
                caminho_salvar = pasta_saida / nome_frame

                if cv2.imwrite(str(caminho_salvar), frame):
                    frames_salvos += 1
                    total_frames += 1
                else:
                    print(f"[AVISO] Falha ao salvar: {caminho_salvar}")

            count += 1

        cap.release()
        print(f"  {video.name}: {frames_salvos} imagens")

    print(f"Total de imagens salvas: {total_frames}")


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent

    extrair_frames_categoria(
        base_dir / "Dataset" / "Normal",
        base_dir / "02_frames_extraidos" / "normais",
        intervalo_frames=3,
    )

    extrair_frames_categoria(
        base_dir / "Dataset" / "Shoplifting",
        base_dir / "02_frames_extraidos" / "suspeitos",
        intervalo_frames=3,
    )
