import os
import numpy as np
import skfuzzy as fuzzy
from skfuzzy import control as ctrl
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

def executar_processamento_fuzzy_completo():
    print("=== INICIANDO PROCESSAMENTO EM LOTE E AVALIAÇÃO DO MOTOR FUZZY ===")
    
    # 1. Definindo o Universo de Discurso
    distancia = ctrl.Antecedent(np.arange(0, 501, 1), 'distancia')
    velocidade = ctrl.Antecedent(np.arange(0, 51, 1), 'velocidade')
    risco = ctrl.Consequent(np.arange(0, 101, 1), 'risco')
    
    # 2. Funções de Pertinência
    distancia['perto'] = fuzzy.trapmf(distancia.universe, [0, 0, 100, 200])
    distancia['media'] = fuzzy.trimf(distancia.universe, [100, 250, 400])
    distancia['longe'] = fuzzy.trapmf(distancia.universe, [300, 400, 500, 500])
    
    velocidade['lenta'] = fuzzy.trapmf(velocidade.universe, [0, 0, 10, 20])
    velocidade['moderada'] = fuzzy.trimf(velocidade.universe, [10, 25, 40])
    velocidade['rapida'] = fuzzy.trapmf(velocidade.universe, [30, 40, 50, 50])
    
    risco['baixo'] = fuzzy.trapmf(risco.universe, [0, 0, 25, 50])
    risco['medio'] = fuzzy.trimf(risco.universe, [25, 50, 75])
    risco['alto'] = fuzzy.trapmf(risco.universe, [50, 75, 100, 100])
    
    # 3. Base de Regras de Inferência Mamdani
    regras = [
        ctrl.Rule(distancia['perto'] & velocidade['rapida'], risco['alto']),
        ctrl.Rule(distancia['perto'] & velocidade['moderada'], risco['medio']),
        ctrl.Rule(distancia['perto'] & velocidade['lenta'], risco['baixo']),
        ctrl.Rule(distancia['media'] & velocidade['rapida'], risco['medio']),
        ctrl.Rule(distancia['media'] & velocidade['moderada'], risco['medio']),
        ctrl.Rule(distancia['media'] & velocidade['lenta'], risco['baixo']),
        ctrl.Rule(distancia['longe'], risco['baixo'])
    ]
    
    sistema_controle = ctrl.ControlSystem(regras)
    
    # 4. Carregamento das Séries Temporais
    path_normal = os.path.abspath('04_dataset_fuzzy/series_fuzzy_normal.npy')
    path_suspeito = os.path.abspath('04_dataset_fuzzy/series_fuzzy_suspeito.npy')
    
    normal_data = np.load(path_normal)
    suspeito_data = np.load(path_suspeito)
    
    y_true = []  # 0 para Normal, 1 para Suspeito
    y_pred = []  # 0 se Risco < 50%, 1 se Risco >= 50%
    riscos_calculados = []
    
    simulador = ctrl.ControlSystemSimulation(sistema_controle)
    
    # Processando amostras normais
    print(f"Processando {len(normal_data)} vetores normais...")
    for sample in normal_data:
        d = np.clip(sample[0], 0, 500)
        v = np.clip(sample[1], 0, 50)
        
        simulador.input['distancia'] = d
        simulador.input['velocidade'] = v
        try:
            simulador.compute()
            r = simulador.output['risco']
        except:
            r = 0.0
            
        riscos_calculados.append(r)
        y_true.append(0)
        y_pred.append(1 if r >= 50.0 else 0)
        
    # Processando amostras suspeitas
    print(f"Processando {len(suspeito_data)} vetores suspeitos...")
    for sample in suspeito_data:
        d = np.clip(sample[0], 0, 500)
        v = np.clip(sample[1], 0, 50)
        
        simulador.input['distancia'] = d
        simulador.input['velocidade'] = v
        try:
            simulador.compute()
            r = simulador.output['risco']
        except:
            r = 0.0
            
        riscos_calculados.append(r)
        y_true.append(1)
        y_pred.append(1 if r >= 50.0 else 0)
        
    # 5. Cálculo das Métricas Finais
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    
    print("\n==================================================")
    print("=== RESULTADOS FINAIS DA AVALIAÇÃO DO MOTOR FUZZY ===")
    print("==================================================")
    print(f"Acurácia Geral:   {acc * 100:.2f}%")
    print(f"Precisão:         {prec * 100:.2f}%")
    print(f"Recall:           {rec * 100:.2f}%")
    print(f"F1-Score:         {f1 * 100:.2f}%")
    print("\nMatriz de Confusão:")
    print(f" [[TN (Normais): {cm[0, 0]}, FP (Falsos Alarmes): {cm[0, 1]}],")
    print(f"  [FN (Omissões): {cm[1, 0]}, TP (Furtos Detectados): {cm[1, 1]}]]")

if __name__ == "__main__":
    executar_processamento_fuzzy_completo()