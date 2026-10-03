# Vigil-IA Data Visualizer MVP

Área isolada para o visualizador web do TCC.

## Executar

Na raiz do projeto:

```powershell
.\.venv_gpu\Scripts\python.exe -m pip install -r .\data_visualizer\requirements.txt
.\.venv_gpu\Scripts\python.exe .\data_visualizer\app.py
```

Abra `http://127.0.0.1:5000`.

Credenciais do MVP:

- usuário: `admin`
- senha: `vigil-ia`

## O que existe

- autenticação por sessão;
- persistência das análises em SQLite (`vigil_ia.sqlite3`);
- biblioteca de vídeos lida de `01_videos_brutos`;
- reprodução de vídeos locais no navegador;
- conversão sob demanda de FMP4 para H.264 compatível com navegadores;
- toggle entre vídeo original e vídeo anotado com boxes, IDs, confiança e pose;
- painel de análises e notificações;
- WebSocket com Flask-SocketIO;
- API `/api/videos`, `/api/analyses` e `/api/analyze`;
- layout responsivo para o MVP.

## Próxima integração

O endpoint `/api/analyze` persiste a análise como `EM_ANALISE`, dispara o `classificar_video.py` em background e atualiza o SQLite com `ANALISADO`, `FURTO`, `NAO_FURTO` e as métricas calculadas. Falhas ficam registradas como `ERRO`.
