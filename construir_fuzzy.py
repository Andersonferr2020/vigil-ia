import os
import numpy as np
import skfuzzy as fuzzy
from skfuzzy import control as ctrl

def construir_sistema_fuzzy():
    print("=== INICIANDO ATIVIDADE 4: CONSTRUÇÃO DO MOTOR FUZZY MAMDANI ===")
    
    # 1. Definindo o Universo de Discurso para cada variável
    # Distância: 0 a 500 pixels (ajustado ao tamanho padrão de entrada)
    distancia = ctrl.Antecedent(np.arange(0, 501, 1), 'distancia')
    
    # Velocidade: 0 a 50 pixels por quadro
    velocidade = ctrl.Antecedent(np.arange(0, 51, 1), 'velocidade')
    
    # Risco: 0 a 100% (Grau de Suspeita)
    risco = ctrl.Consequent(np.arange(0, 101, 1), 'risco')
    
    # 2. Definindo Funções de Pertinência (Fuzzificação)
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
    
    # 4. Criando o Sistema de Controle e Simulação
    sistema_controle = ctrl.ControlSystem(regras)
    simulador = ctrl.ControlSystemSimulation(sistema_controle)
    
    print("Motor Fuzzy Mamdani e Base de Regras construídos com sucesso!")
    
    # 5. Testando nos arquivos .npy criados na Atividade 2
    path_normal = os.path.abspath('04_dataset_fuzzy/series_fuzzy_normal.npy')
    path_suspeito = os.path.abspath('04_dataset_fuzzy/series_fuzzy_suspeito.npy')
    
    if os.path.exists(path_normal) and os.path.exists(path_suspeito):
        normal_data = np.load(path_normal)
        suspeito_data = np.load(path_suspeito)
        
        print(f"\nCarregados {len(normal_data)} vetores normais e {len(suspeito_data)} vetores suspeitos.")
        
        # Teste rápido de inferência em 1 amostra normal
        sample_norm = normal_data[0]
        simulador.input['distancia'] = np.clip(sample_norm[0], 0, 500)
        simulador.input['velocidade'] = np.clip(sample_norm[1], 0, 50)
        simulador.compute()
        risco_norm = simulador.output['risco']
        
        # Teste rápido de inferência em 1 amostra suspeita
        sample_susp = suspeito_data[0]
        simulador.input['distancia'] = np.clip(sample_susp[0], 0, 500)
        simulador.input['velocidade'] = np.clip(sample_susp[1], 0, 50)
        simulador.compute()
        risco_susp = simulador.output['risco']
        
        print(f"Teste Amostra Normal  -> Distância: {sample_norm[0]:.1f}, Velocidade: {sample_norm[1]:.1f} => Risco Calculado: {risco_norm:.2f}%")
        print(f"Teste Amostra Suspeita -> Distância: {sample_susp[0]:.1f}, Velocidade: {sample_susp[1]:.1f} => Risco Calculado: {risco_susp:.2f}%")
    else:
        print("ERRO: Arquivos .npy não encontrados na pasta 04_dataset_fuzzy/")

if __name__ == "__main__":
    construir_sistema_fuzzy()