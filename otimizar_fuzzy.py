import os
import numpy as np
import skfuzzy as fuzzy
from skfuzzy import control as ctrl
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

def otimizar_motor_fuzzy():
    print("=== OTIMIZAÇÃO E CALIBRAÇÃO DO MOTOR FUZZY MAMDANI ===")
    
    # 1. Carregamento das bases .npy
    path_normal = os.path.abspath('04_dataset_fuzzy/series_fuzzy_normal.npy')
    path_suspeito = os.path.abspath('04_dataset_fuzzy/series_fuzzy_suspeito.npy')
    
    if not os.path.exists(path_normal) or not os.path.exists(path_suspeito):
        print("ERRO: Arquivos .npy não encontrados na pasta 04_dataset_fuzzy/")
        return
        
    normal_data = np.load(path_normal)
    suspeito_data = np.load(path_suspeito)
    
    # As colunas são: distância, velocidade e classe.
    if normal_data.shape[1] == 3:
        norm_dist = normal_data[:, 0]
        norm_vel = normal_data[:, 1]
        susp_dist = suspeito_data[:, 0]
        susp_vel = suspeito_data[:, 1]
    else:
        norm_dist = normal_data[:, 0]
        norm_vel = normal_data[:, 1]
        susp_dist = suspeito_data[:, 0]
        susp_vel = suspeito_data[:, 1]

    # 2. Tratamento de Outliers (Percentil 98)
    limit_vel = np.percentile(np.concatenate([norm_vel, susp_vel]), 98)
    limit_dist = np.percentile(np.concatenate([norm_dist, susp_dist]), 98)
    
    norm_dist = np.clip(norm_dist, 0, limit_dist)
    susp_dist = np.clip(susp_dist, 0, limit_dist)
    norm_vel = np.clip(norm_vel, 0, limit_vel)
    susp_vel = np.clip(susp_vel, 0, limit_vel)
    
    # 3. Definindo universos operacionais, limitando outliers extremos.
    limit_dist = min(max(limit_dist, 500.0), 1000.0)
    limit_vel = min(max(limit_vel, 50.0), 1000.0)
    print(f"Limite operacional -> Distância: {limit_dist:.1f} px | Velocidade: {limit_vel:.1f} px/frame")
    distancia = ctrl.Antecedent(np.arange(0, int(limit_dist) + 1, 1), 'distancia')
    velocidade = ctrl.Antecedent(np.arange(0, int(limit_vel) + 1, 1), 'velocidade')
    risco = ctrl.Consequent(np.arange(0, 101, 1), 'risco')

    # 4. Funções de Pertinência Calibradas pela Distribuição
    # Distância (Perto: < 200px | Média: 150 a 450px | Longe: > 400px)
    distancia['perto'] = fuzzy.trapmf(distancia.universe, [0, 0, 100, 200])
    distancia['media'] = fuzzy.trimf(distancia.universe, [100, 250, 400])
    distancia['longe'] = fuzzy.trapmf(distancia.universe, [450, 600, limit_dist, limit_dist])

    # Velocidade (Lenta: < 200 | Moderada: 150 a 500 | Rápida: > 450)
    velocidade['lenta'] = fuzzy.trapmf(velocidade.universe, [0, 0, 100, 250])
    velocidade['moderada'] = fuzzy.trimf(velocidade.universe, [100, 300, 500])
    velocidade['rapida'] = fuzzy.trapmf(velocidade.universe, [500, 700, limit_vel, limit_vel])

    # Risco (Baixo | Médio | Alto)
    risco['baixo'] = fuzzy.trapmf(risco.universe, [0, 0, 25, 50])
    risco['medio'] = fuzzy.trimf(risco.universe, [25, 50, 75])
    risco['alto'] = fuzzy.trapmf(risco.universe, [50, 75, 100, 100])

    # 5. Base de Regras de Inferência Mamdani (SE-ENTÃO)
    regras = [
        ctrl.Rule(distancia['perto'] & velocidade['rapida'], risco['alto']),
        ctrl.Rule(distancia['perto'] & velocidade['moderada'], risco['alto']),
        ctrl.Rule(distancia['perto'] & velocidade['lenta'], risco['medio']),
        ctrl.Rule(distancia['media'] & velocidade['rapida'], risco['alto']),
        ctrl.Rule(distancia['media'] & velocidade['moderada'], risco['medio']),
        ctrl.Rule(distancia['media'] & velocidade['lenta'], risco['baixo']),
        ctrl.Rule(distancia['longe'], risco['baixo'])
    ]

    sistema_controle = ctrl.ControlSystem(regras)
    simulador = ctrl.ControlSystemSimulation(sistema_controle)

    # 6. Processamento em Lote
    y_true = []
    riscos = []

    print("\nCalculando pertinências Fuzzy para a base normal...")
    for d, v in zip(norm_dist, norm_vel):
        simulador.input['distancia'] = d
        simulador.input['velocidade'] = v
        try:
            simulador.compute()
            r = simulador.output['risco']
        except:
            r = 0.0
        riscos.append(r)
        y_true.append(0)

    print("Calculando pertinências Fuzzy para a base suspeita...")
    for d, v in zip(susp_dist, susp_vel):
        simulador.input['distancia'] = d
        simulador.input['velocidade'] = v
        try:
            simulador.compute()
            r = simulador.output['risco']
        except:
            r = 0.0
        riscos.append(r)
        y_true.append(1)

    # 7. Busca do Limiar Ótimo de Risco
    melhor_acc = 0.0
    melhor_threshold = 50.0
    melhor_cm = None
    melhores_metricas = {}

    for t in np.arange(30.0, 75.0, 2.5):
        y_pred = [1 if r >= t else 0 for r in riscos]
        acc = accuracy_score(y_true, y_pred)
        if acc > melhor_acc:
            melhor_acc = acc
            melhor_threshold = t
            melhor_cm = confusion_matrix(y_true, y_pred)
            melhores_metricas = {
                'prec': precision_score(y_true, y_pred),
                'rec': recall_score(y_true, y_pred),
                'f1': f1_score(y_true, y_pred)
            }

    print("\n==================================================")
    print("=== RESULTADOS OTIMIZADOS DO MOTOR FUZZY ===")
    print("==================================================")
    print(f"Limiar Ótimo de Corte: {melhor_threshold:.1f}% de Risco")
    print(f"Acurácia Geral:        {melhor_acc * 100:.2f}%")
    print(f"Precisão:              {melhores_metricas['prec'] * 100:.2f}%")
    print(f"Recall:                {melhores_metricas['rec'] * 100:.2f}%")
    print(f"F1-Score:              {melhores_metricas['f1'] * 100:.2f}%")
    print("\nMatriz de Confusão:")
    print(f" TN (Verdadeiros Normais): {melhor_cm[0, 0]} | FP (Falsos Alarmes):    {melhor_cm[0, 1]}")
    print(f" FN (Omissões):             {melhor_cm[1, 0]} | TP (Furtos Detectados): {melhor_cm[1, 1]}")

if __name__ == "__main__":
    otimizar_motor_fuzzy()