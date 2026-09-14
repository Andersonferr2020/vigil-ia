import os
import numpy as np
import skfuzzy as fuzzy
from skfuzzy import control as ctrl
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

def analisar_e_processar_fuzzy():
    print("=== DIAGNÓSTICO E PROCESSAMENTO CORRIGIDO DO MOTOR FUZZY ===")
    
    # 1. Carregamento dos dados
    path_normal = os.path.abspath('04_dataset_fuzzy/series_fuzzy_normal.npy')
    path_suspeito = os.path.abspath('04_dataset_fuzzy/series_fuzzy_suspeito.npy')
    
    if not os.path.exists(path_normal) or not os.path.exists(path_suspeito):
        print("ERRO: Arquivos .npy não encontrados na pasta 04_dataset_fuzzy/")
        return
        
    normal_data = np.load(path_normal)
    suspeito_data = np.load(path_suspeito)
    
    print(f"Base Normal carregada:   {normal_data.shape} amostras")
    print(f"Base Suspeita carregada: {suspeito_data.shape} amostras")
    
    # Garantir matriz 2D preservando as três colunas: distância, velocidade e classe.
    if normal_data.ndim == 1:
        normal_data = normal_data.reshape(-1, 3)
    if suspeito_data.ndim == 1:
        suspeito_data = suspeito_data.reshape(-1, 3)
    
    # 2. Análise Estatística dos Dados Reais
    norm_dist = normal_data[:, 0]
    norm_vel = normal_data[:, 1]
    susp_dist = suspeito_data[:, 0]
    susp_vel = suspeito_data[:, 1]
    
    print("\n--- ESTATÍSTICAS DOS VETORES EXTRAÍDOS ---")
    print(f"Normais   -> Distância: Média={norm_dist.mean():.1f} | Min={norm_dist.min():.1f} | Max={norm_dist.max():.1f}")
    print(f"Normais   -> Velocidade: Média={norm_vel.mean():.1f} | Min={norm_vel.min():.1f} | Max={norm_vel.max():.1f}")
    print(f"Suspeitos -> Distância: Média={susp_dist.mean():.1f} | Min={susp_dist.min():.1f} | Max={susp_dist.max():.1f}")
    print(f"Suspeitos -> Velocidade: Média={susp_vel.mean():.1f} | Min={susp_vel.min():.1f} | Max={susp_vel.max():.1f}")
    
    # 3. Definição do Universo de Discurso Adaptado
    # Limita os universos aos intervalos definidos para o motor fuzzy.
    max_d = 500.0
    max_v = 50.0
    
    distancia = ctrl.Antecedent(np.arange(0, int(max_d) + 2, 1), 'distancia')
    velocidade = ctrl.Antecedent(np.arange(0, int(max_v) + 2, 1), 'velocidade')
    risco = ctrl.Consequent(np.arange(0, 101, 1), 'risco')
    
    # 4. Funções de Pertinência Fuzzificadas (Mamdani)
    distancia['perto'] = fuzzy.trapmf(distancia.universe, [0, 0, max_d*0.2, max_d*0.4])
    distancia['media'] = fuzzy.trimf(distancia.universe, [max_d*0.25, max_d*0.5, max_d*0.75])
    distancia['longe'] = fuzzy.trapmf(distancia.universe, [max_d*0.6, max_d*0.8, max_d, max_d])
    
    velocidade['lenta'] = fuzzy.trapmf(velocidade.universe, [0, 0, max_v*0.2, max_v*0.4])
    velocidade['moderada'] = fuzzy.trimf(velocidade.universe, [max_v*0.25, max_v*0.5, max_v*0.75])
    velocidade['rapida'] = fuzzy.trapmf(velocidade.universe, [max_v*0.6, max_v*0.8, max_v, max_v])
    
    risco['baixo'] = fuzzy.trapmf(risco.universe, [0, 0, 25, 50])
    risco['medio'] = fuzzy.trimf(risco.universe, [25, 50, 75])
    risco['alto'] = fuzzy.trapmf(risco.universe, [50, 75, 100, 100])
    
    # 5. Base de Regras de Inferência Mamdani
    regras = [
        ctrl.Rule(distancia['perto'] & velocidade['rapida'], risco['alto']),
        ctrl.Rule(distancia['perto'] & velocidade['moderada'], risco['alto']),
        ctrl.Rule(distancia['perto'] & velocidade['lenta'], risco['medio']),
        ctrl.Rule(distancia['media'] & velocidade['rapida'], risco['medio']),
        ctrl.Rule(distancia['media'] & velocidade['moderada'], risco['baixo']),
        ctrl.Rule(distancia['media'] & velocidade['lenta'], risco['baixo']),
        ctrl.Rule(distancia['longe'], risco['baixo'])
    ]
    
    sistema_controle = ctrl.ControlSystem(regras)
    simulador = ctrl.ControlSystemSimulation(sistema_controle)
    
    y_true = []
    y_pred = []
    
    # 6. Processamento Corrigido (Usando sample para distância e sample[1] para velocidade)
    print("\nExecutando simulação Fuzzy em lote...")
    
    for sample in normal_data:
        d = np.clip(sample[0], 0, max_d)
        v = np.clip(sample[1], 0, max_v)
        
        simulador.input['distancia'] = d
        simulador.input['velocidade'] = v
        try:
            simulador.compute()
            r = simulador.output['risco']
        except:
            r = 0.0
            
        y_true.append(0)
        y_pred.append(1 if r >= 50.0 else 0)
        
    for sample in suspeito_data:
        d = np.clip(sample[0], 0, max_d)
        v = np.clip(sample[1], 0, max_v)
        
        simulador.input['distancia'] = d
        simulador.input['velocidade'] = v
        try:
            simulador.compute()
            r = simulador.output['risco']
        except:
            r = 0.0
            
        y_true.append(1)
        y_pred.append(1 if r >= 50.0 else 0)
        
    # 7. Cálculo das Métricas Finais
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    
    print("\n==================================================")
    print("=== RESULTADOS CORRIGIDOS DO MOTOR FUZZY ===")
    print("==================================================")
    print(f"Acurácia Geral:   {acc * 100:.2f}%")
    print(f"Precisão:         {prec * 100:.2f}%")
    print(f"Recall:           {rec * 100:.2f}%")
    print(f"F1-Score:         {f1 * 100:.2f}%")
    print("\nMatriz de Confusão:")
    print(f" TN (Verdadeiros Normais): {cm[0, 0]} | FP (Falsos Alarmes):    {cm[0, 1]}")
    print(f" FN (Omissões):             {cm[1, 0]} | TP (Furtos Detectados): {cm[1, 1]}")

if __name__ == "__main__":
    analisar_e_processar_fuzzy()