from pathlib import Path
import csv
import tkinter as tk
from tkinter import messagebox, ttk
from PIL import Image, ImageDraw, ImageTk

ROOT = Path(__file__).resolve().parent
REPORT = ROOT / 'auditoria_rotulos' / 'simulacao_limpeza.csv'
DECISIONS = ROOT / 'auditoria_rotulos' / 'revisao_manual_iou90.csv'
LABELS = ROOT / 'labels_backup_original' / 'train'
IMAGES = ROOT / 'dataset_yolo_final' / 'images' / 'train'


def read_csv(path):
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def image_path(stem):
    for ext in ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'):
        path = IMAGES / (stem + ext)
        if path.exists():
            return path
    return None


class Reviewer:
    def __init__(self, root):
        self.root = root
        root.title('VIGIL-IA | Revisao IoU > 90% | Antes / Depois')
        root.geometry('1100x720')
        root.minsize(820, 580)
        self.items = read_csv(REPORT)
        if not self.items:
            raise RuntimeError('O relatorio de simulacao esta vazio.')
        self.decisions = {}
        if DECISIONS.exists():
            for row in read_csv(DECISIONS):
                key = (row['arquivo'], row['linha_mantida'], row['linha_candidata_remocao'])
                self.decisions[key] = row['decisao']
        self.index = next((i for i, item in enumerate(self.items)
                           if self.key(item) not in self.decisions), 0)
        self.status = tk.StringVar()
        self.detail = tk.StringVar()
        self.pending = tk.StringVar(value='')
        self.photos = [None, None]
        self.original = None
        self.lines = []
        self.error = None

        ttk.Label(root, textvariable=self.status, font=('Segoe UI', 12, 'bold')).pack(pady=(10, 4))
        ttk.Label(root, textvariable=self.detail).pack(pady=(0, 7))
        self.panels = ttk.Frame(root)
        self.panels.pack(fill='both', expand=True, padx=10)
        self.labels = []
        self.label_texts = []
        for title in ('ANTES | Anotacoes originais', 'DEPOIS | Previa sem a candidata'):
            frame = ttk.Frame(self.panels)
            frame.pack(side='left', fill='both', expand=True, padx=5)
            ttk.Label(frame, text=title, font=('Segoe UI', 11, 'bold')).pack(pady=4)
            label = tk.Label(frame, bg='#202020', fg='white', anchor='center')
            label.pack(fill='both', expand=True)
            self.labels.append(label)
            ttk.Label(frame, text='Rotulos YOLO ' + ('originais' if not self.label_texts else 'apos simulacao'),
                      font=('Segoe UI', 9, 'bold')).pack(anchor='w', pady=(6, 2))
            text_frame = ttk.Frame(frame)
            text_frame.pack(fill='x')
            label_text = tk.Text(text_frame, height=6, wrap='none',
                                 font=('Consolas', 9), bg='#f6f6f6', fg='#202020')
            label_text.pack(side='left', fill='x', expand=True)
            scrollbar = ttk.Scrollbar(text_frame, orient='vertical', command=label_text.yview)
            scrollbar.pack(side='right', fill='y')
            label_text.configure(yscrollcommand=scrollbar.set, state='disabled')
            label_text.tag_configure('keep', foreground='#008d49')
            label_text.tag_configure('candidate', foreground='#ce2424')
            label_text.tag_configure('other', foreground='#1769aa')
            self.label_texts.append(label_text)

        ttk.Label(root, text='Verde: manter | Vermelho: candidata | Azul: outras bolsas/mochilas | Cinza: pessoa').pack(pady=5)
        options = ttk.Frame(root)
        options.pack(pady=5)
        for text, value in (('Sim (S) - remover', 'sim'),
                            ('Nao (N) - manter', 'nao'),
                            ('Duvida (D)', 'duvida')):
            ttk.Radiobutton(options, text=text, variable=self.pending, value=value,
                            command=self.update_preview).pack(side='left', padx=12)
        actions = ttk.Frame(root)
        actions.pack(pady=(3, 12))
        ttk.Button(actions, text='Anterior', command=self.previous).pack(side='left', padx=6)
        ttk.Button(actions, text='Confirmar decisao', command=self.confirm).pack(side='left', padx=6)
        ttk.Button(actions, text='Proximo', command=self.next).pack(side='left', padx=6)
        ttk.Button(actions, text='Salvar captura PNG', command=self.save_screenshot).pack(side='left', padx=6)

        root.bind('<KeyPress-s>', lambda e: self.select('sim'))
        root.bind('<KeyPress-n>', lambda e: self.select('nao'))
        root.bind('<KeyPress-d>', lambda e: self.select('duvida'))
        root.bind('<Left>', lambda e: self.previous())
        root.bind('<Right>', lambda e: self.next())
        root.bind('<Return>', lambda e: self.confirm())
        root.bind('<Configure>', self.on_resize)
        self.resize_job = None
        self.show()

    @staticmethod
    def key(item):
        return (item['arquivo'], item['linha_mantida'], item['linha_candidata_remocao'])

    def show(self):
        item = self.items[self.index]
        reviewed = sum(self.key(x) in self.decisions for x in self.items)
        current = self.decisions.get(self.key(item), 'pendente')
        self.status.set(f'Imagem {self.index + 1}/{len(self.items)} | Revisadas: {reviewed} | Decisao salva: {current}')
        self.detail.set(f"{item['arquivo']} | manter linha {item['linha_mantida']} | candidata linha {item['linha_candidata_remocao']}")
        self.pending.set(self.decisions.get(self.key(item), ''))
        self.error = None
        self.original = None
        self.lines = []
        path = image_path(Path(item['arquivo']).stem)
        label_path = LABELS / item['arquivo']
        if path is None:
            self.error = 'IMAGEM NAO ENCONTRADA'
        elif not label_path.exists():
            self.error = 'ROTULO ORIGINAL NAO ENCONTRADO'
        else:
            try:
                with Image.open(path) as source:
                    self.original = source.convert('RGB')
                self.lines = label_path.read_text(encoding='utf-8').splitlines()
                keep = int(item['linha_mantida'])
                remove = int(item['linha_candidata_remocao'])
                if not (1 <= keep <= len(self.lines) and 1 <= remove <= len(self.lines)):
                    self.error = 'NUMERO DE LINHA INVALIDO'
            except (OSError, ValueError) as exc:
                self.error = str(exc)
        self.update_label_texts()
        self.update_preview()

    def update_label_texts(self):
        if self.error or not self.lines:
            for widget in self.label_texts:
                widget.configure(state='normal')
                widget.delete('1.0', 'end')
                widget.insert('end', self.error or 'Sem rotulos disponiveis')
                widget.configure(state='disabled')
            return
        item = self.items[self.index]
        keep = int(item['linha_mantida'])
        remove = int(item['linha_candidata_remocao'])
        for after, widget in enumerate(self.label_texts):
            widget.configure(state='normal')
            widget.delete('1.0', 'end')
            for number, line in enumerate(self.lines, 1):
                if after and number == remove:
                    continue
                mark = ' [MANTER]' if number == keep else ' [CANDIDATA]' if number == remove else ''
                tag = 'keep' if number == keep else 'candidate' if number == remove else 'other'
                widget.insert('end', f'L{number}: {line}{mark}\n', tag)
            widget.configure(state='disabled')

    def draw_image(self, after):
        im = self.original.copy()
        w, h = im.size
        draw = ImageDraw.Draw(im)
        item = self.items[self.index]
        keep = int(item['linha_mantida'])
        remove = int(item['linha_candidata_remocao'])
        # O painel DEPOIS sempre simula a remocao; a escolha ainda nao altera o arquivo.
        for number, line in enumerate(self.lines, 1):
            if after and number == remove:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                cls = int(parts[0])
                xc, yc, bw, bh = map(float, parts[1:5])
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
            x1 = int((xc - bw / 2) * w)
            y1 = int((yc - bh / 2) * h)
            x2 = int((xc + bw / 2) * w)
            y2 = int((yc + bh / 2) * h)
            draw.rectangle((x1, y1, x2, y2), outline=color, width=max(2, w // 250))
            ty = max(0, y1 - 18) if number != remove else min(h - 18, y2 + 2)
            draw.text((max(0, x1), ty), f'L{number} {name}', fill=color,
                      stroke_width=1, stroke_fill='black')
        return im

    def update_preview(self):
        if self.error or self.original is None:
            for label in self.labels:
                label.configure(image='', text=self.error or 'Sem imagem')
            return
        for idx, after in enumerate((False, True)):
            image = self.draw_image(after)
            width = max(250, self.labels[idx].winfo_width() - 12)
            height = max(120, min(390, self.labels[idx].winfo_height() - 12))
            image.thumbnail((width, height), Image.Resampling.LANCZOS)
            self.photos[idx] = ImageTk.PhotoImage(image)
            self.labels[idx].configure(image=self.photos[idx], text='')

    def on_resize(self, event):
        if event.widget is self.root:
            if self.resize_job is not None:
                self.root.after_cancel(self.resize_job)
            self.resize_job = self.root.after(180, self.update_preview)

    def select(self, value):
        self.pending.set(value)
        self.update_preview()

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

    def confirm(self):
        if self.error:
            messagebox.showwarning('Revisao', 'Nao e possivel confirmar: ' + self.error)
            return
        decision = self.pending.get()
        if decision not in ('sim', 'nao', 'duvida'):
            messagebox.showwarning('Revisao', 'Selecione Sim, Nao ou Duvida antes de confirmar.')
            return
        self.decisions[self.key(self.items[self.index])] = decision
        self.save()
        if self.index < len(self.items) - 1:
            self.index += 1
        else:
            messagebox.showinfo('Revisao', 'Ultima imagem alcancada. Decisao salva.')
        self.show()

    def previous(self):
        if self.index > 0:
            self.index -= 1
            self.show()

    def next(self):
        if self.index < len(self.items) - 1:
            self.index += 1
            self.show()

    def save_screenshot(self):
        if self.error or self.original is None:
            messagebox.showwarning('Captura', 'Imagem indisponivel.')
            return
        before = self.draw_image(False)
        after = self.draw_image(True)
        max_side = 1100
        for im in (before, after):
            im.thumbnail((max_side, 850), Image.Resampling.LANCZOS)
        pad, header = 20, 45
        out = Image.new('RGB', (before.width + after.width + pad * 3,
                                max(before.height, after.height) + header + pad * 2), 'white')
        out.paste(before, (pad, header))
        out.paste(after, (before.width + pad * 2, header))
        draw = ImageDraw.Draw(out)
        draw.text((pad, 12), 'ANTES - Rotulos originais', fill='black')
        draw.text((before.width + pad * 2, 12), 'DEPOIS - Previa sem candidata', fill='black')
        folder = ROOT / 'auditoria_rotulos' / 'capturas_iou90'
        folder.mkdir(parents=True, exist_ok=True)
        name = Path(self.items[self.index]['arquivo']).stem
        destination = folder / f'{self.index + 1:03d}_{name}_antes_depois.png'
        out.save(destination)
        messagebox.showinfo('Captura', f'PNG salvo em:\n{destination}')


if __name__ == '__main__':
    if not REPORT.exists():
        raise SystemExit(f'Relatorio nao encontrado: {REPORT}')
    root = tk.Tk()
    Reviewer(root)
    root.mainloop()
