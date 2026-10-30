from pathlib import Path
from collections import defaultdict
import csv
import re
import tkinter as tk
from tkinter import ttk, messagebox

import cv2
from PIL import Image, ImageTk

ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "Dataset"
YOLO = ROOT / "dataset_yolo_final"
OUTPUT = ROOT / "auditoria_rotulos" / "revisao_videos_mochilas.csv"

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

FRAME_PATTERN = re.compile(r"^(normal|shoplifting|suspeito)_(.+)_f\d+$", re.I)


def video_key(category, name):
    return f"{category.lower()}_{name.lower()}"


def find_videos():
    videos = []

    for category in ("Normal", "Shoplifting"):
        folder = DATASET / category

        if not folder.exists():
            continue

        for path in sorted(folder.iterdir(), key=lambda p: p.name.lower()):
            if path.is_file() and path.suffix.lower() in VIDEO_EXTS:
                videos.append({
                    "path": path,
                    "category": category,
                    "key": video_key(category, path.stem),
                })

    return videos


def scan_dataset():
    stats = defaultdict(lambda: {
        "images": 0,
        "labeled_images": 0,
        "boxes": 0,
        "splits": set(),
    })

    for split in ("train", "val", "test"):
        image_dir = YOLO / "images" / split
        label_dir = YOLO / "labels" / split

        if not image_dir.exists():
            continue

        for image in image_dir.iterdir():
            if image.suffix.lower() not in IMAGE_EXTS:
                continue

            match = FRAME_PATTERN.match(image.stem)

            if not match:
                continue

            category, original_name = match.groups()
            if category.lower() == "suspeito":
                category = "shoplifting"
            key = video_key(category, original_name)

            item = stats[key]
            item["images"] += 1
            item["splits"].add(split)

            label_path = label_dir / (image.stem + ".txt")

            if not label_path.exists():
                continue

            count = 0

            for line in label_path.read_text(
                encoding="utf-8-sig"
            ).splitlines():
                parts = line.strip().split()

                if parts and parts[0] == "1":
                    count += 1

            if count:
                item["labeled_images"] += 1
                item["boxes"] += count

    return stats


def load_decisions():
    decisions = {}

    if not OUTPUT.exists():
        return decisions

    with OUTPUT.open(
        "r", encoding="utf-8-sig", newline=""
    ) as file:
        for row in csv.DictReader(file):
            decisions[row["video_key"]] = row

    return decisions



class VideoReviewer:
    def __init__(self, root):
        self.root = root
        root.title("VIGIL-IA | Auditoria de v?deos")

        largura = min(1000, root.winfo_screenwidth() - 60)
        altura = min(700, root.winfo_screenheight() - 110)
        root.geometry(f"{max(480, largura)}x{max(400, altura)}")
        root.minsize(420, 350)

        self.videos = find_videos()
        self.stats = scan_dataset()
        self.decisions = load_decisions()

        if not self.videos:
            raise RuntimeError("Nenhum v?deo encontrado.")

        self.index = next(
            (i for i, v in enumerate(self.videos)
             if v["key"] not in self.decisions), 0
        )

        self.cap = None
        self.playing = False
        self.photo = None
        self.last_frame = None
        self.frame_index = 0
        self.frame_count = 0
        self.fps = 25.0

        self.status = tk.StringVar()
        self.details = tk.StringVar()
        self.progress = tk.StringVar()
        self.answer = tk.StringVar()
        self.notice = tk.StringVar()

        # Regi?o superior: informa??es e navega??o
        top = ttk.Frame(root, padding=(8, 5))
        top.pack(side="top", fill="x")

        ttk.Label(
            top, textvariable=self.status,
            font=("Segoe UI", 11, "bold")
        ).pack()

        ttk.Label(
            top, textvariable=self.details,
            wraplength=850
        ).pack(pady=2)

        navigation = ttk.Frame(top)
        navigation.pack(pady=3)

        for text, command in (
            ("Anterior", self.previous),
            ("-5 s", lambda: self.seek(-5)),
            ("In?cio", self.restart_video),
            ("Pausar / Reproduzir", self.toggle_play),
            ("+5 s", lambda: self.seek(5)),
            ("Pr?ximo", self.next),
        ):
            ttk.Button(
                navigation, text=text, command=command
            ).pack(side="left", padx=2)

        # Regi?o inferior: respostas e salvamento.
        # Empacotada antes do v?deo para reservar seu espa?o.
        bottom = ttk.Frame(root, padding=(8, 5))
        bottom.pack(side="bottom", fill="x")

        ttk.Label(
            bottom,
            text="Existe mochila ou bolsa em algum momento do v?deo?",
            font=("Segoe UI", 10, "bold")
        ).pack()

        answers = ttk.Frame(bottom)
        answers.pack(pady=4)

        for text, value in (
            ("SIM", "sim"),
            ("N?O", "nao"),
            ("D?VIDA", "duvida"),
        ):
            ttk.Radiobutton(
                answers, text=text,
                variable=self.answer, value=value
            ).pack(side="left", padx=15)

        ttk.Button(
            bottom,
            text="SALVAR E PR?XIMO V?DEO",
            command=self.save
        ).pack(fill="x", padx=30, pady=4)

        ttk.Label(
            bottom, textvariable=self.notice,
            foreground="#1769aa"
        ).pack()

        # Regi?o central: apenas o v?deo se expande.
        center = ttk.Frame(root, padding=(8, 2))
        center.pack(side="top", fill="both", expand=True)

        self.video_label = tk.Label(
            center, bg="#151515", fg="white",
            text="Carregando v?deo..."
        )
        self.video_label.pack(fill="both", expand=True)

        ttk.Label(
            center, textvariable=self.progress
        ).pack()

        root.bind("<Control-Return>", lambda e: self.save())
        root.protocol("WM_DELETE_WINDOW", self.close)

        self.open_video()
        root.after(30, self.update_frame)

    def current(self):
        return self.videos[self.index]

    def open_video(self):
        if self.cap is not None:
            self.cap.release()

        video = self.current()
        self.cap = cv2.VideoCapture(str(video["path"]))

        self.playing = self.cap.isOpened()
        self.frame_index = 0
        self.last_frame = None

        if not self.playing:
            self.notice.set("Erro ao abrir o v?deo.")
            return

        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        if self.fps <= 0:
            self.fps = 25.0

        self.frame_count = int(
            self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        saved = self.decisions.get(video["key"], {})
        self.answer.set(saved.get("resposta", ""))

        stat = self.stats.get(video["key"], {})
        images = stat.get("images", 0)
        labeled = stat.get("labeled_images", 0)
        boxes = stat.get("boxes", 0)
        splits = ", ".join(sorted(stat.get("splits", set())))

        reviewed = sum(
            v["key"] in self.decisions for v in self.videos
        )

        self.status.set(
            f"V?deo {self.index + 1}/{len(self.videos)} | "
            f"Respondidos: {reviewed} | "
            f"{video['category']} / {video['path'].name}"
        )

        self.details.set(
            f"Imagens: {images} | "
            f"Com r?tulo de mochila: {labeled} | "
            f"Caixas: {boxes} | "
            f"Conjuntos: {splits or 'n?o encontrado'}"
        )

        self.notice.set("")
        self.progress.set("0 s")

    def draw_frame(self, frame):
        self.last_frame = frame

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width = frame.shape[:2]

        available_w = max(self.video_label.winfo_width() - 8, 1)
        available_h = max(self.video_label.winfo_height() - 8, 1)

        scale = min(
            available_w / width,
            available_h / height
        )

        new_size = (
            max(1, int(width * scale)),
            max(1, int(height * scale))
        )

        image = Image.fromarray(frame).resize(
            new_size, Image.Resampling.BILINEAR
        )

        self.photo = ImageTk.PhotoImage(image)
        self.video_label.configure(image=self.photo, text="")

    def update_frame(self):
        if self.playing and self.cap is not None:
            success, frame = self.cap.read()

            if success:
                self.frame_index += 1
                self.draw_frame(frame)

                elapsed = self.frame_index / self.fps
                total = self.frame_count / self.fps

                self.progress.set(
                    f"{elapsed:.1f} s / {total:.1f} s"
                )
            else:
                self.playing = False
                self.notice.set(
                    "V?deo finalizado. Responda e salve."
                )

        self.root.after(
            max(15, int(1000 / self.fps)),
            self.update_frame
        )

    def toggle_play(self):
        if self.cap is not None:
            self.playing = not self.playing

    def seek(self, seconds):
        if self.cap is None:
            return

        target = self.frame_index + int(seconds * self.fps)
        target = max(
            0, min(target, max(0, self.frame_count - 1))
        )

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, target)
        self.frame_index = target
        self.playing = True

    def restart_video(self):
        if self.cap is not None:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.frame_index = 0
            self.playing = True

    def save(self):
        response = self.answer.get()

        if response not in ("sim", "nao", "duvida"):
            messagebox.showwarning(
                "Resposta necess?ria",
                "Selecione SIM, N?O ou D?VIDA."
            )
            return

        video = self.current()
        stat = self.stats.get(video["key"], {})
        labeled = stat.get("labeled_images", 0)

        if response == "nao" and labeled > 0:
            status = "revisar_rotulos_existentes"
        elif response == "sim" and labeled == 0:
            status = "revisar_rotulos_ausentes"
        elif response == "duvida":
            status = "revisao_manual"
        else:
            status = "sem_divergencia_global"

        self.decisions[video["key"]] = {
            "video_key": video["key"],
            "categoria": video["category"],
            "video": video["path"].name,
            "resposta": response,
            "imagens": stat.get("images", 0),
            "imagens_com_rotulo": labeled,
            "caixas_mochila": stat.get("boxes", 0),
            "status": status,
        }

        OUTPUT.parent.mkdir(parents=True, exist_ok=True)

        fields = [
            "video_key", "categoria", "video",
            "resposta", "imagens", "imagens_com_rotulo",
            "caixas_mochila", "status",
        ]

        with OUTPUT.open(
            "w", encoding="utf-8-sig", newline=""
        ) as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            for key in sorted(self.decisions):
                writer.writerow(self.decisions[key])

        if self.index < len(self.videos) - 1:
            self.index += 1
            self.open_video()
        else:
            self.notice.set(
                "?ltimo v?deo alcan?ado. Resposta salva."
            )
            messagebox.showinfo(
                "Fim da lista",
                "?ltimo v?deo alcan?ado."
            )

    def previous(self):
        if self.index > 0:
            self.index -= 1
            self.open_video()

    def next(self):
        if self.index < len(self.videos) - 1:
            self.index += 1
            self.open_video()

    def close(self):
        if self.cap is not None:
            self.cap.release()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()

    try:
        VideoReviewer(root)
        root.mainloop()
    except Exception as exc:
        messagebox.showerror(
            "Erro no programa", str(exc)
        )
        root.destroy()
