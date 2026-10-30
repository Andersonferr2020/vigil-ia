# VIGIL-IA

Sistema academico experimental de visao computacional para analise de comportamentos potencialmente suspeitos envolvendo bolsas e mochilas.

Uma classificacao de suspeita nao constitui prova de furto.

## Componentes

- Detector YOLO M4: classes person e backpack_bag.
- Classificador Original: classificar_video.py.
- Classificador V2: classificar_video_v2.py.
- Modelo de pose: yolov8n-pose.pt.
- Dashboard local: data_visualizer/.
- Historico experimental: resultados M1, M2, M3 e M4.

## Dataset de videos

- Dataset/Normal: 90 videos.
- Dataset/Shoplifting: 92 videos.
- Total: 182 videos.

Fonte informada: Mohd. Aquib Ansari e Dushyant Kumar Singh, Mendeley Data.
https://data.mendeley.com/datasets/r3yjf35hzr/1
DOI: 10.17632/r3yjf35hzr.1

## Dataset de treinamento M4

| Conjunto | Imagens | Rotulos |
|---|---:|---:|
| Treinamento | 14.023 | 14.023 |
| Validacao | 2.975 | 2.975 |
| Teste | 3.298 | 3.298 |
| Total | 20.296 | 20.296 |

Diretorio: dataset_yolo_revisado/
Configuracao: dataset_yolo_revisado/data.yaml

## Modelos

- Detector atual: runs/detect/treino_vigilia_M4_revisado/weights/best.pt
- Pose: yolov8n-pose.pt
- Pesos antigos M1-M3 nao fazem parte da versao executavel atual.

## Ambiente de referencia

- Windows e Python 3.12.6.
- Ultralytics 8.4.181.
- PyTorch 2.5.1+cu121 e CUDA 12.1.
- Dependencias principais: requirements.txt.

## Execucao

Na raiz do projeto, com o ambiente Python configurado:

Original: python classificar_video.py CAMINHO_DO_VIDEO

V2: python classificar_video_v2.py CAMINHO_DO_VIDEO

Dashboard: python data_visualizer/app.py

Acesso local: http://127.0.0.1:5000

Consulte data_visualizer/README.md para instrucoes adicionais.

## Limitacoes e licenciamento

O sistema e experimental e pode produzir falsos positivos e falsos negativos.
O dashboard e um MVP local, nao preparado para exposicao publica.
A licenca do codigo ainda sera definida.
As condicoes de redistribuicao dos videos, imagens e pesos devem ser verificadas separadamente.

## Procedencia e atribuicao dos dados

Os 182 videos originais foram obtidos do dataset publicado por Mohd. Aquib Ansari e Dushyant Kumar Singh no Mendeley Data.

Fonte: https://data.mendeley.com/datasets/r3yjf35hzr/1
DOI: https://doi.org/10.17632/r3yjf35hzr.1
Licenca informada na fonte original: Creative Commons Attribution 4.0 International (CC BY 4.0).
Texto da licenca: https://creativecommons.org/licenses/by/4.0/

As 20.296 imagens do dataset M4 foram extraidas exclusivamente desses 182 videos.
Os 20.296 arquivos de rotulos correspondentes representam anotacoes utilizadas no treinamento e avaliacao do detector M4.

O dataset M4 constitui uma preparacao e anotacao dos dados originais para treinamento YOLO.
A redistribuicao deve preservar a atribuicao aos autores, a referencia a fonte, a licenca aplicavel e a indicacao das adaptacoes realizadas.
A licenca dos dados nao substitui a definicao de uma licenca propria para o codigo do projeto.
