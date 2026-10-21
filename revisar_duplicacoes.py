from pathlib import Path
import csv
import tkinter as tk
from tkinter import messagebox, ttk

try:
    from PIL import Image, ImageDraw, ImageFont, ImageTk
except ImportError:
    raise SystemExit('Instale Pillow: python -m pip install pillow')

ROOT = Path(__file__).resolve().parent
REPORT = ROOT / 'auditoria_rotulos' / 'simulacao_limpeza.csv'
DECISIONS = ROOT / 'auditoria_rotulos' / 'revisao_manual.csv'
LABELS = ROOT / 'labels_revisados' / 'train'
IMAGES = ROOT / 'dataset_yolo_final' / 'images' / 'train'


def read_csv(path):
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def image_path(stem):
    for ext in ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'):
        p = IMAGES / (stem + ext)
        if p.exists():
            return p
    return None


class Reviewer:
    def __init__(self, root):
        self.root = root
        self.root.title('VIGIL-IA | Revisao de duplicacoes')
        self.items = read_csv(REPORT)
        if not self.items:
            raise RuntimeError('O relatorio de simulacao esta vazio.')
        self.decisions = {}
        if DECISIONS.exists():
            for row in read_csv(DECISIONS):
                self.decisions[(row['arquivo'], row['linha_mantida'], row['linha_candidata_remocao'])] = row['decisao']
        self.index = next((i for i, item in enumerate(self.items) if self.key(item) not in self.decisions), 0)
        self.photo = None
        self.status = tk.StringVar()
        self.detail = tk.StringVar()
        ttk.Label(root, textvariable=self.status, font=('Segoe UI', 12, 'bold')).pack(padx=12, pady=(10, 4))
        ttk.Label(root, textvariable=self.detail).pack(padx=12, pady=(0, 6))
        self.canvas = tk.Label(root, background='#202020')
        self.canvas.pack(padx=12, pady=4)
        bar = ttk.Frame(root)
        bar.pack(pady=10)
        ttk.Button(bar, text='Sim (S)', command=lambda: self.answer('sim')).pack(side='left', padx=5)
        ttk.Button(bar, text='Nao (N)', command=lambda: self.answer('nao')).pack(side='left', padx=5)
        ttk.Button(bar, text='Duvida (D)', command=lambda: self.answer('duvida')).pack(side='left', padx=5)
        ttk.Button(bar, text='Anterior (←)', command=self.previous).pack(side='left', padx=5)
        ttk.Button(bar, text='Proximo (→)', command=self.next).pack(side='left', padx=5)
        ttk.Label(root, text='Verde: manter | Vermelho: candidato | Azul: outras bolsas/mochilas | Cinza: pessoa').pack(pady=(0, 12))
        root.bind('<KeyPress-s>', lambda e: self.answer('sim'))
        root.bind('<KeyPress-n>', lambda e: self.answer('nao'))
        root.bind('<KeyPress-d>', lambda e: self.answer('duvida'))
        root.bind('<Left>', lambda e: self.previous())
        root.bind('<Right>', lambda e: self.next())
        self.show()

    @staticmethod
    def key(item):
        return (item['arquivo'], item['linha_mantida'], item['linha_candidata_remocao'])

    def show(self):
        item = self.items[self.index]
        reviewed = sum(self.key(x) in self.decisions for x in self.items)
        previous = self.decisions.get(self.key(item), 'pendente')
        self.status.set(f'Imagem {self.index + 1}/{len(self.items)} | Revisadas: {reviewed} | Decisao: {previous}')
        self.detail.set(f"{item['arquivo']}  |  manter linha {item['linha_mantida']}  |  candidata linha {item['linha_candidata_remocao']}")
        path = image_path(Path(item['arquivo']).stem)
        if path is None:
            self.canvas.configure(text='IMAGEM NAO ENCONTRADA', image='', foreground='white', width=80, height=25)
            return
        with Image.open(path) as src:
            im = src.convert('RGB')
        w, h = im.size
        draw = ImageDraw.Draw(im)
        label_path = LABELS / item['arquivo']
        if not label_path.exists():
            self.canvas.configure(text='ROTULO NAO ENCONTRADO', image='', foreground='white', width=80, height=25)
            return
        lines = label_path.read_text(encoding='utf-8').splitlines()
        keep = int(item['linha_mantida'])
        remove = int(item['linha_candidata_remocao'])
        for number, line in enumerate(lines, start=1):
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                cls = int(parts[0]); xc, yc, bw, bh = map(float, parts[1:5])
            except ValueError:
                continue
            if number == keep:
                color, name = '#00e060', 'MANTER'
            elif number == remove:
                color, name = '#ff4040', 'CANDIDATA'
            elif cls == 1:
                color, name = '#3aa9ff', 'OUTRA BOLSA/MOCHILA'
            else:
                color, name = '#b0b0b0', 'PESSOA' if cls == 0 else f'CLASSE {cls}'
            x1 = int((xc - bw / 2) * w); y1 = int((yc - bh / 2) * h)
            x2 = int((xc + bw / 2) * w); y2 = int((yc + bh / 2) * h)
            draw.rectangle((x1, y1, x2, y2), outline=color, width=max(2, w // 250))
            # Rotulos em posicoes diferentes para evitar que as etiquetas se escondam.
            ty = max(0, y1 - 18) if number != remove else min(h - 18, y2 + 2)
            draw.text((max(0, x1), ty), f'L{number} {name}', fill=color, stroke_width=1, stroke_fill='black')
        im.thumbnail((1100, 740), Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(im)
        self.canvas.configure(image=self.photo, text='', width=im.width, height=im.height)

    def save(self):
        DECISIONS.parent.mkdir(parents=True, exist_ok=True)
        fields = ['arquivo', 'linha_mantida', 'linha_candidata_remocao', 'decisao']
        temp = DECISIONS.with_suffix('.tmp')
        with temp.open('w', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for item in self.items:
                key = self.key(item)
                if key in self.decisions:
                    writer.writerow(dict(zip(fields, (*key, self.decisions[key]))))
        temp.replace(DECISIONS)

    def answer(self, decision):
        self.decisions[self.key(self.items[self.index])] = decision
        self.save()
        if self.index < len(self.items) - 1:
            self.index += 1
        else:
            messagebox.showinfo('Revisao', 'Ultima imagem alcancada. As decisoes foram salvas.')
        self.show()

    def previous(self):
        if self.index > 0:
            self.index -= 1
            self.show()

    def next(self):
        if self.index < len(self.items) - 1:
            self.index += 1
            self.show()


if __name__ == '__main__':
    if not REPORT.exists():
        raise SystemExit(f'Relatorio nao encontrado: {REPORT}')
    root = tk.Tk()
    Reviewer(root)
    root.mainloop()
