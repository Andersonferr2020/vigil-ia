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
- biblioteca de vídeos lida de `Dataset/Normal` e `Dataset/Shoplifting`;
- reprodução de vídeos locais no navegador;
- conversão sob demanda de FMP4 para H.264 compatível com navegadores;
- toggle entre vídeo original e vídeo anotado com boxes, IDs, confiança e pose;
- painel de análises e notificações;
- WebSocket com Flask-SocketIO;
- API `/api/videos`, `/api/analyses` e `/api/analyze`;
- layout responsivo para o MVP.

## Próxima integração

O endpoint `/api/analyze` persiste a análise como `EM_ANALISE`, dispara o `classificar_video.py` em background e atualiza o SQLite com `ANALISADO`, `FURTO`, `NAO_FURTO` e as métricas calculadas. Falhas ficam registradas como `ERRO`.

## Configuracao por variaveis de ambiente

O dashboard foi desenvolvido para uso local.

- VIGILIA_SECRET_KEY: chave de sessao Flask.
- VIGILIA_USER: usuario de acesso.
- VIGILIA_PASSWORD: senha de acesso.

As variaveis sao opcionais. Sem elas, o sistema utiliza as credenciais demonstrativas documentadas acima.

Para configurar valores proprios no PowerShell:

    $env:VIGILIA_SECRET_KEY = "uma-chave-aleatoria-longa"
    $env:VIGILIA_USER = "meu_usuario"
    $env:VIGILIA_PASSWORD = "minha_senha"
    .\.venv_gpu\Scripts\python.exe .\data_visualizer\app.py

Seguranca: o dashboard e um MVP local, nao preparado para exposicao direta na internet.