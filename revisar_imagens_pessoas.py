
import csv
import json
import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path
from datetime import datetime

import cv2
from PIL import Image, ImageTk

BASE = Path("auditoria_rotulos")
FILA = BASE / "fila_revisao_pessoas_3729.csv"
DESTINO = BASE / "rotulos_revisados_pessoas"
PROGRESSO = BASE / "progresso_revisao_pessoas.json"
LOG = BASE / "historico_edicoes_pessoas.csv"

DESTINO.mkdir(parents=True, exist_ok=True)

with FILA.open(encoding="utf-8-sig", newline="") as f:
    ITENS = list(csv.DictReader(f))

if not ITENS:
    raise SystemExit("A fila de revis\u25b6o est\u25b6 vazia.")

class Editor:
    def __init__(self, root):
        self.root = root
        self.root.title("VIGIL-IA | Editor de Rótulos")
        self.root.geometry("1100x780")
        self.root.minsize(680, 480)

        self.indice = 0
        self.caixas = []
        self.selecionada = None
        self.modo = "selecionar"
        self.inicio = None
        self.arraste = None
        self.operacao = None
        self.caixa_inicial = None
        self.alterado = False
        self.imagem = None
        self.foto = None
        self.escala = 1
        self.ox = 0
        self.oy = 0
        self.img_w = 1
        self.img_h = 1
        self.concluidos = set()

        if PROGRESSO.exists():
            try:
                dados = json.loads(PROGRESSO.read_text(encoding="utf-8"))
                self.indice = min(
                    max(0, int(dados.get("indice", 0))),
                    len(ITENS) - 1
                )
                self.concluidos = set(dados.get("concluidos", []))
            except (ValueError, TypeError):
                pass

        topo = ttk.Frame(root, padding=8)
        topo.pack(fill="x")

        ttk.Button(topo, text="\u25b6 Anterior", command=self.anterior).pack(
            side="left", padx=3
        )
        ttk.Button(topo, text="Próxima \u25b6", command=self.proxima).pack(
            side="left", padx=3
        )
        ttk.Button(topo, text="Salvar", command=self.salvar).pack(
            side="left", padx=3
        )
        ttk.Button(
            topo, text="Salvar e próxima",
            command=self.salvar_proxima
        ).pack(side="left", padx=3)

        self.info = ttk.Label(topo, text="")
        self.info.pack(side="right", padx=8)

        self.canvas = tk.Canvas(
            root, background="#20242a",
            highlightthickness=0, cursor="crosshair"
        )
        self.canvas.pack(fill="both", expand=True, padx=8, pady=4)

        self.canvas.bind("<Configure>", lambda e: self.desenhar())
        self.canvas.bind("<ButtonPress-1>", self.mouse_down)
        self.canvas.bind("<B1-Motion>", self.mouse_move)
        self.canvas.bind("<ButtonRelease-1>", self.mouse_up)

        ferramentas = ttk.Frame(root, padding=8)
        ferramentas.pack(fill="x")

        ttk.Button(
            ferramentas, text="Selecionar / Ajustar",
            command=lambda: self.definir_modo("selecionar")
        ).pack(side="left", padx=3)

        ttk.Button(
            ferramentas, text="Adicionar pessoa",
            command=lambda: self.definir_modo("adicionar")
        ).pack(side="left", padx=3)

        ttk.Button(
            ferramentas, text="Excluir caixa",
            command=self.excluir
        ).pack(side="left", padx=3)

        ttk.Button(
            ferramentas, text="Marcar revisada",
            command=self.marcar_revisada
        ).pack(side="left", padx=3)

        self.status = ttk.Label(root, padding=(12, 4))
        self.status.pack(fill="x")

        self.root.bind("<Delete>", lambda e: self.excluir())
        self.root.bind("<space>", lambda e: self.proxima())
        self.root.bind("<Control-s>", lambda e: self.salvar())
        self.root.bind("<Right>", lambda e: self.proxima())
        self.root.bind("<Left>", lambda e: self.anterior())
        self.root.protocol("WM_DELETE_WINDOW", self.fechar)

        self.carregar()

    def item(self):
        return ITENS[self.indice]

    def caminho_saida(self):
        item = self.item()
        return DESTINO / item["conjunto"] / Path(item["rotulo"]).name

    def carregar(self):
        item = self.item()
        caminho = Path(item["imagem"])

        self.imagem = cv2.imread(str(caminho))
        if self.imagem is None:
            messagebox.showerror("Erro", f"Imagem n\u25b6o encontrada:\n{caminho}")
            return

        self.img_h, self.img_w = self.imagem.shape[:2]

        rotulo = self.caminho_saida()
        if not rotulo.exists():
            fontes = {
                "etapa_1": BASE / "rotulos_revisados",
                "etapa_2": BASE / "rotulos_revisados_ausentes",
                "etapa_3": BASE / "rotulos_revisados_multiplas_mochilas",
            }
            pasta = fontes[item["etapa_origem"]]
            candidato = pasta / item["conjunto"] / Path(item["rotulo"]).name
            rotulo = candidato if candidato.exists() else Path(item["rotulo"])

        self.caixas = []
        for linha in rotulo.read_text(encoding="utf-8-sig").splitlines():
            partes = linha.split()
            if len(partes) >= 5:
                try:
                    classe = int(float(partes[0]))
                    valores = [float(x) for x in partes[1:5]]
                    self.caixas.append([classe, *valores])
                except ValueError:
                    continue

        self.selecionada = None
        self.alterado = False
        self.definir_modo("selecionar")
        self.atualizar_info()
        self.desenhar()

    def atualizar_info(self):
        item = self.item()
        chave = item["conjunto"] + "/" + Path(item["imagem"]).name
        revisada = "REVISADA" if chave in self.concluidos else "PENDENTE"

        self.info.config(
            text=f"{self.indice + 1}/{len(ITENS)} | {item['conjunto']} | {revisada}"
        )
        self.status.config(
            text=(
                f"{Path(item['imagem']).name} | "
                f"Caixas de mochila: "
                f"{sum(c[0] == 0 for c in self.caixas)} | "
                f"Modo: {self.modo}"
            )
        )

    def definir_modo(self, modo):
        self.modo = modo
        self.atualizar_info()

    def desenhar(self):
        if self.imagem is None:
            return

        cw = max(self.canvas.winfo_width(), 1)
        ch = max(self.canvas.winfo_height(), 1)

        self.escala = min(cw / self.img_w, ch / self.img_h)
        nw = max(1, int(self.img_w * self.escala))
        nh = max(1, int(self.img_h * self.escala))

        self.ox = (cw - nw) // 2
        self.oy = (ch - nh) // 2

        img = cv2.cvtColor(self.imagem, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img).resize((nw, nh))
        self.foto = ImageTk.PhotoImage(img)

        self.canvas.delete("all")
        self.canvas.create_image(
            self.ox, self.oy, anchor="nw", image=self.foto
        )

        for i, (classe, cx, cy, w, h) in enumerate(self.caixas):
            x1 = (cx - w / 2) * self.img_w * self.escala + self.ox
            y1 = (cy - h / 2) * self.img_h * self.escala + self.oy
            x2 = (cx + w / 2) * self.img_w * self.escala + self.ox
            y2 = (cy + h / 2) * self.img_h * self.escala + self.oy

            cor = "#ffb000" if classe == 1 else "#37d67a"
            largura = 3 if i == self.selecionada else 2

            self.canvas.create_rectangle(
                x1, y1, x2, y2, outline=cor, width=largura
            )
            self.canvas.create_text(
                x1 + 3, y1 + 3,
                text="mochila" if classe == 1 else "pessoa",
                anchor="nw", fill=cor
            )

            if i == self.selecionada and classe == 0:
                for x in (x1, x2):
                    for y in (y1, y2):
                        self.canvas.create_rectangle(
                            x-5, y-5, x+5, y+5,
                            fill="white", outline=cor
                        )

        if self.inicio and self.arraste and self.modo == "adicionar":
            self.canvas.create_rectangle(
                *self.inicio, *self.arraste,
                outline="#37d67a", width=2, dash=(5, 3)
            )

    def coordenadas(self, x, y):
        ix = (x - self.ox) / (self.img_w * self.escala)
        iy = (y - self.oy) / (self.img_h * self.escala)
        return max(0, min(1, ix)), max(0, min(1, iy))

    def encontrar(self, x, y):
        px, py = self.coordenadas(x, y)
        for i in reversed(range(len(self.caixas))):
            classe, cx, cy, w, h = self.caixas[i]
            if classe != 0:
                continue
            if cx-w/2 <= px <= cx+w/2 and cy-h/2 <= py <= cy+h/2:
                return i
        return None

    def limites(self, caixa):
        _, cx, cy, w, h = caixa
        return [cx-w/2, cy-h/2, cx+w/2, cy+h/2]

    def atualizar_caixa(self, indice, x1, y1, x2, y2):
        x1, x2 = sorted((max(0, min(1, x1)), max(0, min(1, x2))))
        y1, y2 = sorted((max(0, min(1, y1)), max(0, min(1, y2))))
        if x2-x1 < 0.003 or y2-y1 < 0.003:
            return
        self.caixas[indice] = [
            0, (x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1
        ]
        self.alterado = True

    def mouse_down(self, evento):
        self.inicio = (evento.x, evento.y)
        self.arraste = None
        self.operacao = None

        if self.modo != "selecionar":
            return

        px, py = self.coordenadas(evento.x, evento.y)
        tolerancia = 9 / max(
            1, min(self.img_w, self.img_h) * self.escala
        )

        if self.selecionada is not None:
            caixa = self.caixas[self.selecionada]
            if caixa[0] == 0:
                x1, y1, x2, y2 = self.limites(caixa)
                cantos = {
                    "nw": (x1, y1),
                    "ne": (x2, y1),
                    "sw": (x1, y2),
                    "se": (x2, y2),
                }
                for nome, (x, y) in cantos.items():
                    if abs(px-x) <= tolerancia and abs(py-y) <= tolerancia:
                        self.operacao = nome
                        self.caixa_inicial = caixa[:]
                        return

        self.selecionada = self.encontrar(evento.x, evento.y)

        if self.selecionada is not None:
            self.operacao = "mover"
            self.caixa_inicial = self.caixas[self.selecionada][:]

        self.desenhar()

    def mouse_move(self, evento):
        if self.inicio is None:
            return

        if self.modo == "adicionar":
            self.arraste = (evento.x, evento.y)
            self.desenhar()
            return

        if self.selecionada is None or self.operacao is None:
            return

        px0, py0 = self.coordenadas(*self.inicio)
        px, py = self.coordenadas(evento.x, evento.y)
        _, cx, cy, w, h = self.caixa_inicial

        if self.operacao == "mover":
            novo_cx = max(w/2, min(1-w/2, cx + px-px0))
            novo_cy = max(h/2, min(1-h/2, cy + py-py0))
            self.caixas[self.selecionada] = [
                0, novo_cx, novo_cy, w, h
            ]
            self.alterado = True
        else:
            x1, y1, x2, y2 = self.limites(self.caixa_inicial)
            if "w" in self.operacao:
                x1 = min(px, x2-0.003)
            if "e" in self.operacao:
                x2 = max(px, x1+0.003)
            if "n" in self.operacao:
                y1 = min(py, y2-0.003)
            if "s" in self.operacao:
                y2 = max(py, y1+0.003)
            self.atualizar_caixa(
                self.selecionada, x1, y1, x2, y2
            )

        self.desenhar()

    def mouse_up(self, evento):
        if self.modo == "adicionar" and self.inicio:
            x1, y1 = self.coordenadas(*self.inicio)
            x2, y2 = self.coordenadas(evento.x, evento.y)

            x1, x2 = sorted((x1, x2))
            y1, y2 = sorted((y1, y2))

            if x2-x1 > 0.005 and y2-y1 > 0.005:
                self.caixas.append([
                    0, (x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1
                ])
                self.alterado = True

        self.inicio = None
        self.arraste = None
        self.operacao = None
        self.atualizar_info()
        self.desenhar()

    def excluir(self):
        if self.selecionada is None:
            messagebox.showinfo(
                "Selecionar pessoa",
                "Clique primeiro na caixa da pessoa que deseja excluir."
            )
            return

        if self.caixas[self.selecionada][0] != 0:
            return

        del self.caixas[self.selecionada]
        self.selecionada = None
        self.alterado = True
        self.atualizar_info()
        self.desenhar()

    def salvar(self):
        saida = self.caminho_saida()
        saida.parent.mkdir(parents=True, exist_ok=True)

        conteudo = "\n".join(
            f"{int(c[0])} " + " ".join(f"{v:.6f}" for v in c[1:])
            for c in self.caixas
        )

        saida.write_text(
            conteudo + ("\n" if conteudo else ""),
            encoding="utf-8"
        )

        if self.alterado:
            novo = not LOG.exists()
            with LOG.open("a", encoding="utf-8-sig", newline="") as f:
                writer = csv.writer(f)
                if novo:
                    writer.writerow([
                        "data", "imagem", "conjunto",
                        "caixas_mochila_apos_edicao"
                    ])
                writer.writerow([
                    datetime.now().isoformat(timespec="seconds"),
                    self.item()["imagem"],
                    self.item()["conjunto"],
                    sum(c[0] == 0 for c in self.caixas)
                ])

        self.alterado = False
        self.salvar_progresso()
        self.atualizar_info()

    def salvar_progresso(self):
        PROGRESSO.write_text(
            json.dumps({
                "indice": self.indice,
                "concluidos": sorted(self.concluidos)
            }, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )

    def marcar_revisada(self):
        self.salvar()
        chave = (
            self.item()["conjunto"] + "/" +
            Path(self.item()["imagem"]).name
        )
        self.concluidos.add(chave)
        self.salvar_progresso()
        self.atualizar_info()
        self.proxima()

    def salvar_proxima(self):
        self.salvar()
        self.proxima()

    def proxima(self):
        if self.alterado:
            self.salvar()
        if self.indice < len(ITENS) - 1:
            self.indice += 1
            self.salvar_progresso()
            self.carregar()

    def anterior(self):
        if self.alterado:
            self.salvar()
        if self.indice > 0:
            self.indice -= 1
            self.salvar_progresso()
            self.carregar()

    def fechar(self):
        if self.alterado:
            self.salvar()
        self.root.destroy()

root = tk.Tk()

from tkinter import font

for nome in ("TkDefaultFont", "TkTextFont", "TkMenuFont",
             "TkHeadingFont", "TkCaptionFont"):
    try:
        fonte = font.nametofont(nome)
        fonte.configure(family="Segoe UI", size=10)
    except tk.TclError:
        pass

Editor(root)
root.mainloop()
