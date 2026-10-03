import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from classificar_video import (  # noqa: E402
    IMG_SIZE,
    associar_ids,
    centro_caixa,
    distancia_esqueleto_pessoa,
    escolher_punho,
    localizar_modelo,
    obter_deteccoes,
    obter_esqueletos,
    obter_punhos,
)


def gerar_video_marcado(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target
    model = YOLO(localizar_modelo())
    pose_model = YOLO("yolo11n-pose.pt")
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"Não foi possível abrir o vídeo: {source}")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    writer = cv2.VideoWriter(
        str(temporary),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError("Não foi possível criar o vídeo anotado.")

    scale_x = IMG_SIZE / max(width, 1)
    scale_y = IMG_SIZE / max(height, 1)
    previous_centers = {}
    next_id = 1
    frame_number = 0

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame_number += 1
            detection = model(frame, imgsz=IMG_SIZE, device=0, verbose=False)[0]
            pose = pose_model(frame, imgsz=IMG_SIZE, device=0, verbose=False)[0]
            people, bags = obter_deteccoes(detection)
            skeletons = obter_esqueletos(pose)
            centers, next_id = associar_ids(people, previous_centers, next_id)
            previous_centers = centers

            annotated = pose.plot(
                img=frame,
                boxes=False,
                labels=False,
                conf=False,
                kpt_radius=2,
                kpt_line=True,
            )

            for bag_id, bag in enumerate(bags, start=1):
                x1, y1, x2, y2 = bag.xyxy[0].cpu().numpy().astype(int)
                confidence = float(bag.conf.item())
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 165, 0), 2)
                cv2.putText(
                    annotated,
                    f"{bag_id} - Bolsa - {confidence * 100:.0f}%",
                    (x1, max(y1 - 8, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 165, 0),
                    2,
                )

            for person_id, person in zip(centers, people):
                x1, y1, x2, y2 = person.xyxy[0].cpu().numpy().astype(int)
                confidence = float(person.conf.item())
                person_center = centro_caixa(person)
                skeleton = min(
                    skeletons,
                    key=lambda item: distancia_esqueleto_pessoa(item, person_center),
                    default=None,
                )
                wrists = obter_punhos(skeleton) if skeleton is not None else []
                nearest_bag = min(bags, key=lambda bag: np.linalg.norm(centro_caixa(bag) - person_center), default=None)
                if nearest_bag is not None and wrists:
                    wrist = escolher_punho(wrists, centro_caixa(nearest_bag))
                    wrist_xy = tuple(np.round(wrist).astype(int))
                    bag_xy = tuple(np.round(centro_caixa(nearest_bag)).astype(int))
                    cv2.line(annotated, wrist_xy, bag_xy, (255, 255, 0), 2)
                    cv2.circle(annotated, wrist_xy, 4, (255, 0, 255), -1)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    annotated,
                    f"{person_id} - Pessoa - {confidence * 100:.0f}%",
                    (x1, max(y1 - 8, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

            cv2.rectangle(annotated, (0, 0), (width, 48), (18, 16, 10), -1)
            cv2.putText(
                annotated,
                f"VIGIL-IA | FRAME {frame_number}",
                (18, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (242, 197, 0),
                2,
            )
            writer.write(annotated)
    finally:
        capture.release()
        writer.release()

    return temporary
