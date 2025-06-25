import pandas as pd
import numpy as np
np.random.seed(42)

n = 200

df = pd.DataFrame({
    'user_id': np.arange(1, n+1),
    'dt_yearmonth': pd.date_range('2023-01-01', periods=n, freq='MS').strftime('%Y-%m'),
    'idade': np.random.randint(22, 60, n),
    'tempo_empresa': np.random.uniform(0.5, 20, n),
    'flag_previdencia': np.random.randint(0, 2, n),
    'flag_estabilidade': np.random.randint(0, 2, n),
    'flag_promocao_merito_engaj_ult_12m': np.random.randint(0, 2, n),
    'nota_performance_ult_ano': np.random.randint(1, 6, n),
    'nota_performance_perfil_ano': np.random.randint(1, 6, n),
    'qtd_horas_extras_pagas_ma12': np.random.uniform(0, 100, n),
    'qtd_horas_extras_pagas_ma3': np.random.uniform(0, 30, n),
    'qtd_sobrecarga_pago_ma12': np.random.uniform(0, 50, n),
    'qtd_sobrecarga_pago_ma3': np.random.uniform(0, 15, n),
    'estado_residencia': np.random.choice(['SP','RJ','MG','RS'], n),
    'estado_civil': np.random.choice(['Solteiro','Casado','Divorciado'], n),
    'subgrupo_piramide_atual': np.random.choice(['A','B','C'], n),
    'estado_civil_alteracao_ult_12m': np.random.choice(['Sim','Não'], n),
    'flag_desligamento_voluntario': np.random.randint(0, 2, n),
    'flag_pool': np.random.randint(0, 2, n),
    'pred_flag_desligamento_voluntario': np.random.randint(0, 2, n)
})

df.to_csv('df_ficticio.csv', index=False)
