from pathlib import Path
import cv2

base = Path(__file__).resolve().parent / "Dataset"
extensoes = {".mp4", ".avi", ".mov", ".mkv"}

total_videos = 0
total_frames_estimados = 0
total_segundos = 0
falhas = []

for pasta in [base / "Normal", base / "Shoplifting"]:
    print(f"\nCategoria: {pasta.name}")

    videos = sorted(
        p for p in pasta.iterdir()
        if p.is_file() and p.suffix.lower() in extensoes
    )

    frames_categoria = 0

    for video in videos:
        cap = cv2.VideoCapture(str(video))

        if not cap.isOpened():
            falhas.append(str(video))
            continue

        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        if frames <= 0:
            falhas.append(str(video))
            continue

        estimados = (frames + 2) // 3
        segundos = frames / fps if fps > 0 else 0

        total_videos += 1
        total_frames_estimados += estimados
        frames_categoria += estimados
        total_segundos += segundos

    print(f"Vídeos lidos: {len(videos)}")
    print(f"Frames estimados a cada 3 quadros: {frames_categoria:,}")

print("\n===== RESUMO =====")
print(f"Vídeos lidos com sucesso: {total_videos}")
print(f"Frames estimados: {total_frames_estimados:,}")
print(f"Duração total aproximada: {total_segundos / 3600:.2f} horas")
print(f"Falhas de leitura: {len(falhas)}")

if falhas:
    print("\nArquivos com problemas:")
    for arquivo in falhas:
        print(arquivo)
