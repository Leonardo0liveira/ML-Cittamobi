
# ═══════════════════════════════════════════════════════════════════════════
# CÓDIGO DE EXEMPLO - INFERÊNCIA MODELO V7 ENSEMBLE FINAL
# ═══════════════════════════════════════════════════════════════════════════

import joblib
import lightgbm as lgb
import xgboost as xgb
import pandas as pd
import numpy as np
import json

# 1. CARREGAR MODELOS E ARTEFATOS
print("Carregando modelos...")
lgb_model = lgb.Booster(model_file='lightgbm_model_v7_FINAL.txt')
xgb_model = xgb.Booster()
xgb_model.load_model('xgboost_model_v7_FINAL.json')
scaler = joblib.load('scaler_v7_FINAL.pkl')

with open('selected_features_v7_FINAL.txt', 'r') as f:
    selected_features = [line.strip() for line in f]

with open('model_config_v7_FINAL.json', 'r') as f:
    config = json.load(f)

print(f"✓ Modelos carregados")
print(f"✓ Features: {len(selected_features)}")
print(f"✓ Threshold: {config['ensemble']['threshold']}")
print(f"✓ Pesos: LightGBM={config['ensemble']['weights']['lightgbm']:.3f}, XGBoost={config['ensemble']['weights']['xgboost']:.3f}")

# 2. PREPARAR DADOS (exemplo com 1 registro)
# Substitua isso pelos seus dados reais
new_data = pd.DataFrame({
    # ... adicione suas features aqui ...
    # Certifique-se de ter todas as {len(selected_features)} features
})

# Verificar se todas as features estão presentes
missing_features = set(selected_features) - set(new_data.columns)
if missing_features:
    print(f"⚠️  Features faltando: {missing_features}")
    # Adicionar features faltantes com valor 0 ou calcular
    for feat in missing_features:
        new_data[feat] = 0

# Selecionar e ordenar features
new_data_selected = new_data[selected_features]

# 3. NORMALIZAR
new_data_scaled = scaler.transform(new_data_selected)

# 4. PREDIÇÃO
# Predição LightGBM
y_proba_lgb = lgb_model.predict(new_data_scaled)

# Predição XGBoost
dmatrix = xgb.DMatrix(new_data_selected)
y_proba_xgb = xgb_model.predict(dmatrix)

# Ensemble (média ponderada)
w_lgb = config['ensemble']['weights']['lightgbm']
w_xgb = config['ensemble']['weights']['xgboost']
y_proba_ensemble = w_lgb * y_proba_lgb + w_xgb * y_proba_xgb

# Classificação final
threshold = config['ensemble']['threshold']
y_pred = (y_proba_ensemble >= threshold).astype(int)

print(f"\n=== RESULTADO ===")
print(f"Probabilidade de conversão: {y_proba_ensemble[0]:.4f} ({y_proba_ensemble[0]*100:.2f}%)")
print(f"Predição: {'CONVERSÃO' if y_pred[0] == 1 else 'NÃO CONVERSÃO'}")

# ═══════════════════════════════════════════════════════════════════════════
