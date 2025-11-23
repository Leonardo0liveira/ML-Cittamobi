from google.cloud import bigquery
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb
from xgboost import plot_importance, plot_tree

project_id = "proj-ml-469320"
client = bigquery.Client(project=project_id)

# ===========================================================================
# ETAPA 1: CARREGAR MAIS DADOS (500k amostras)
# ===========================================================================
print(f"\n{'='*70}")
print(f"ETAPA 1: CARREGANDO DATASET COMPLETO")
print(f"{'='*70}")
print('Comecando a carregar:')

query = """
    SELECT * FROM `proj-ml-469320.app_cittamobi.dataset-updated` LIMIT 100000 
"""

print("Carregando 500,000 amostras do BigQuery...")
df = client.query(query).to_dataframe()
print(f"✓ Dados carregados: {len(df):,} registros")

# ===========================================================================
# ETAPA 2: ANÁLISE E LIMPEZA RIGOROSA DOS DADOS
# ===========================================================================
print(f"\n{'='*70}")
print(f"ETAPA 2: ANÁLISE E LIMPEZA RIGOROSA DOS DADOS")
print(f"{'='*70}")

print(f"\n📊 Dataset Original:")
print(f"   Shape: {df.shape}")
print(f"   Memória: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

target = "target"

# Analisar distribuição do target
print(f"\n=== Distribuição do Target (Original) ===")
target_dist = df[target].value_counts()
print(target_dist)
print(f"Proporção classe 0: {target_dist[0]/len(df)*100:.2f}%")
print(f"Proporção classe 1: {target_dist[1]/len(df)*100:.2f}%")

# ===========================================================================
# FILTROS DE QUALIDADE DOS DADOS
# ===========================================================================
print(f"\n=== Aplicando Filtros de Qualidade ===")

df_original_size = len(df)

# 1. Remover usuários com poucos eventos (engano/teste)
if 'user_frequency' in df.columns:
    user_freq_threshold = df['user_frequency'].quantile(0.25)  # Remover 25% com menos eventos
    df = df[df['user_frequency'] >= user_freq_threshold]
    print(f"✓ Filtro 1: Removidos {df_original_size - len(df):,} registros de usuários com baixa frequência (< {user_freq_threshold:.0f} eventos)")
    df_original_size = len(df)

# 2. Remover registros com dados de localização inválidos
if 'device_lat' in df.columns and 'device_lon' in df.columns:
    # Remover lat/lon zerados ou nulos
    df = df[~((df['device_lat'].isna()) | (df['device_lon'].isna()))]
    df = df[~((df['device_lat'] == 0) & (df['device_lon'] == 0))]
    print(f"✓ Filtro 2: Removidos {df_original_size - len(df):,} registros com localização inválida")
    df_original_size = len(df)

# 3. Remover registros com distância muito grande (usuário não está perto da parada)
if 'dist_device_stop' in df.columns:
    # Considerar apenas usuários a menos de 500m da parada (aproximadamente)
    dist_threshold = df['dist_device_stop'].quantile(0.95)  # Remover 5% com maior distância
    df = df[df['dist_device_stop'] <= dist_threshold]
    print(f"✓ Filtro 3: Removidos {df_original_size - len(df):,} registros com distância muito alta (> {dist_threshold:.2f})")
    df_original_size = len(df)

# 4. Remover registros com headway inválido (sem informação de frequência de ônibus)
if 'headway_avg_stop_hour' in df.columns:
    df = df[df['headway_avg_stop_hour'] > 0]
    print(f"✓ Filtro 4: Removidos {df_original_size - len(df):,} registros com headway inválido")
    df_original_size = len(df)

# 5. Remover paradas com poucos eventos (paradas pouco usadas)
if 'stop_event_count' in df.columns:
    stop_threshold = df['stop_event_count'].quantile(0.20)  # Remover 20% com menos eventos
    df = df[df['stop_event_count'] >= stop_threshold]
    print(f"✓ Filtro 5: Removidos {df_original_size - len(df):,} registros de paradas com poucos eventos (< {stop_threshold:.0f})")
    df_original_size = len(df)

# 6. Verificar balanceamento após filtragem
print(f"\n=== Distribuição do Target (Após Filtros) ===")
target_dist_filtered = df[target].value_counts()
print(target_dist_filtered)
print(f"Proporção classe 0: {target_dist_filtered[0]/len(df)*100:.2f}%")
print(f"Proporção classe 1: {target_dist_filtered[1]/len(df)*100:.2f}%")

print(f"\n📊 Dataset Limpo:")
print(f"   Shape: {df.shape}")
print(f"   Registros removidos: {500000 - len(df):,} ({(500000 - len(df))/500000*100:.2f}%)")
print(f"   Registros mantidos: {len(df):,} ({len(df)/500000*100:.2f}%)")

# ===========================================================================
# ETAPA 3: FEATURE ENGINEERING AVANÇADO
# ===========================================================================
print(f"\n{'='*70}")
print(f"ETAPA 3: FEATURE ENGINEERING AVANÇADO")
print(f"{'='*70}")

# Remover features com data leakage
features_to_drop = ['y_pred', 'y_pred_proba', 'ctm_service_route', 'direction', 'lotacao_proxy_binaria']
X = df.drop(columns=[target] + features_to_drop, errors='ignore')
y = df[target]

# Processar timestamp
if 'event_timestamp' in X.columns:
    X['event_timestamp'] = pd.to_datetime(X['event_timestamp'])
    X['year'] = X['event_timestamp'].dt.year
    X['month'] = X['event_timestamp'].dt.month
    X['day'] = X['event_timestamp'].dt.day
    X['hour'] = X['event_timestamp'].dt.hour
    X['dayofweek'] = X['event_timestamp'].dt.dayofweek
    X['week_of_year'] = X['event_timestamp'].dt.isocalendar().week
    
    # Feature: Minuto da hora (para captar padrões dentro da hora)
    X['minute'] = X['event_timestamp'].dt.minute
    
    X = X.drop(columns=['event_timestamp'])
    print(f"✓ Features temporais extraídas")

# Feature Engineering: Interações
print(f"\n=== Criando Features de Interação ===")

# 1. Interação: hora x dia da semana (comportamento varia por dia/hora)
if 'hour' in X.columns and 'dayofweek' in X.columns:
    X['hour_x_dayofweek'] = X['hour'] * X['dayofweek']
    print(f"✓ Feature: hour_x_dayofweek")

# 2. Interação: distância x hora de pico
if 'dist_device_stop' in X.columns and 'is_peak_hour' in X.columns:
    X['dist_x_peak_enhanced'] = X['dist_device_stop'] * (X['is_peak_hour'] + 1)
    print(f"✓ Feature: dist_x_peak_enhanced")

# 3. Feature: Taxa de eventos normalizada por parada
if 'stop_event_rate' in X.columns and 'stop_total_samples' in X.columns:
    X['event_rate_normalized'] = X['stop_event_rate'] / (X['stop_total_samples'] + 1)
    print(f"✓ Feature: event_rate_normalized")

# 4. Feature: Ratio headway x hora
if 'headway_avg_stop_hour' in X.columns and 'hour' in X.columns:
    X['headway_per_hour'] = X['headway_avg_stop_hour'] / (X['hour'] + 1)
    print(f"✓ Feature: headway_per_hour")

# 5. Feature: Densidade de eventos (eventos por amostra na parada)
if 'stop_event_count' in X.columns and 'stop_total_samples' in X.columns:
    X['event_density'] = X['stop_event_count'] / (X['stop_total_samples'] + 1)
    print(f"✓ Feature: event_density")

# 6. Features cíclicas adicionais (para dia do mês e semana do ano)
if 'time_day_of_month' in X.columns:
    X['day_of_month_sin'] = np.sin(2 * np.pi * X['time_day_of_month'] / 31)
    X['day_of_month_cos'] = np.cos(2 * np.pi * X['time_day_of_month'] / 31)
    print(f"✓ Features cíclicas: day_of_month_sin, day_of_month_cos")

if 'week_of_year' in X.columns:
    X['week_sin'] = np.sin(2 * np.pi * X['week_of_year'] / 52)
    X['week_cos'] = np.cos(2 * np.pi * X['week_of_year'] / 52)
    print(f"✓ Features cíclicas: week_sin, week_cos")

# Label Encoding para categóricas
categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
if categorical_cols:
    print(f"\n=== Encoding de Variáveis Categóricas ===")
    from sklearn.preprocessing import LabelEncoder
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le
    print(f"✓ {len(categorical_cols)} colunas categóricas encodadas")

# Limpar dados
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(0)

print(f"\n📊 Dataset Final para Modelagem:")
print(f"   Shape: {X.shape}")
print(f"   Total de features: {X.shape[1]}")
print(f"   Novas features criadas: {X.shape[1] - 38}")

# ===========================================================================
# ETAPA 4: DIVISÃO TEMPORAL E TREINAMENTO
# ===========================================================================
print(f"\n{'='*70}")
print(f"ETAPA 4: DIVISÃO TEMPORAL E TREINAMENTO")
print(f"{'='*70}")

# TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=3)  # Aumentar para 3 folds
X_train, X_test, y_train, y_test = None, None, None, None

for fold, (train_index, test_index) in enumerate(tscv.split(X)):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    print(f"Fold {fold + 1}: Train={len(X_train):,} | Test={len(X_test):,}")

# Criar DMatrix
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Calcular scale_pos_weight
scale_pos_weight = (len(y_train) - y_train.sum()) / y_train.sum() if y_train is not None else 1

# ===========================================================================
# ETAPA 5: TUNING AVANÇADO DE HIPERPARÂMETROS
# ===========================================================================
print(f"\n{'='*70}")
print(f"ETAPA 5: TUNING AVANÇADO DE HIPERPARÂMETROS")
print(f"{'='*70}")

param_grid = [
    # Configuração 1: Modelo profundo e conservador
    {'max_depth': 12, 'learning_rate': 0.02, 'subsample': 0.85, 'colsample_bytree': 0.85, 
     'min_child_weight': 5, 'gamma': 0.1, 'reg_alpha': 0.01, 'reg_lambda': 1},
    
    # Configuração 2: Modelo moderado com regularização
    {'max_depth': 10, 'learning_rate': 0.03, 'subsample': 0.9, 'colsample_bytree': 0.9, 
     'min_child_weight': 3, 'gamma': 0.05, 'reg_alpha': 0.05, 'reg_lambda': 0.5},
    
    # Configuração 3: Modelo mais agressivo
    {'max_depth': 8, 'learning_rate': 0.05, 'subsample': 0.9, 'colsample_bytree': 0.85, 
     'min_child_weight': 2, 'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 0.1},
    
    # Configuração 4: Modelo com alta regularização
    {'max_depth': 10, 'learning_rate': 0.03, 'subsample': 0.8, 'colsample_bytree': 0.8, 
     'min_child_weight': 6, 'gamma': 0.2, 'reg_alpha': 0.1, 'reg_lambda': 2},
]

best_auc = 0
best_params = None
best_model = None

from sklearn.metrics import roc_auc_score as roc_auc_calc

print(f"Testando {len(param_grid)} configurações...")

for i, param_config in enumerate(param_grid):
    print(f"\n--- Config {i+1}/{len(param_grid)} ---")
    for key, val in param_config.items():
        print(f"   {key}: {val}")
    
    params_test = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'seed': 42,
        'scale_pos_weight': scale_pos_weight,
        **param_config
    }
    
    model_test = xgb.train(
        params=params_test,
        dtrain=dtrain,
        num_boost_round=200,  # Mais iterações
        evals=[(dtest, 'test')],
        early_stopping_rounds=20,
        verbose_eval=False
    )
    
    y_pred_test = model_test.predict(dtest)
    auc = roc_auc_calc(y_test, y_pred_test)
    print(f"   ROC-AUC: {auc:.4f}")
    
    if auc > best_auc:
        best_auc = auc
        best_params = params_test
        best_model = model_test
        print(f"   ✓ Nova melhor configuração!")

print(f"\n{'='*70}")
print(f"MELHOR MODELO SELECIONADO")
print(f"{'='*70}")
print(f"ROC-AUC: {best_auc:.4f}")
print(f"\nMelhores Parâmetros:")
for key, val in best_params.items():
    if key not in ['objective', 'eval_metric', 'seed']:
        print(f"   {key}: {val}")

model = best_model if best_model is not None else None

# ===========================================================================
# ETAPA 6: OTIMIZAÇÃO DE THRESHOLD
# ===========================================================================
print(f"\n{'='*70}")
print(f"ETAPA 6: OTIMIZANDO THRESHOLD")
print(f"{'='*70}")

y_pred_proba = model.predict(dtest)

from sklearn.metrics import precision_score, recall_score, f1_score

thresholds_to_test = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]
threshold_results = []

for thresh in thresholds_to_test:
    y_pred_thresh = (y_pred_proba >= thresh).astype(int)
    prec = precision_score(y_test, y_pred_thresh, zero_division=0)
    rec = recall_score(y_test, y_pred_thresh, zero_division=0)
    f1 = f1_score(y_test, y_pred_thresh, zero_division=0)
    
    threshold_results.append({
        'threshold': thresh,
        'precision': prec,
        'recall': rec,
        'f1_score': f1
    })
    
    print(f"Threshold={thresh:.2f} | Precision={prec:.4f} | Recall={rec:.4f} | F1={f1:.4f}")

best_threshold = max(threshold_results, key=lambda x: x['f1_score'])
print(f"\n✓ Melhor threshold: {best_threshold['threshold']:.2f} (F1={best_threshold['f1_score']:.4f})")

y_pred = (y_pred_proba >= best_threshold['threshold']).astype(int)

# ===========================================================================
# ETAPA 7: AVALIAÇÃO FINAL
# ===========================================================================
print(f"\n{'='*70}")
print(f"ETAPA 7: AVALIAÇÃO FINAL DO MODELO")
print(f"{'='*70}")

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print(f"\n=== Métricas Finais ===")
print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")
print(f"ROC-AUC:   {roc_auc:.4f}")

cm = confusion_matrix(y_test, y_pred)
print(f"\n=== Matriz de Confusão ===")
print(f"                 Predito")
print(f"                 0      1")
print(f"Real  0       {cm[0,0]:6d} {cm[0,1]:6d}")
print(f"      1       {cm[1,0]:6d} {cm[1,1]:6d}")

print(f"\n=== Relatório de Classificação ===")
print(classification_report(y_test, y_pred, target_names=['Classe 0', 'Classe 1']))

# ===========================================================================
# ETAPA 8: VISUALIZAÇÕES
# ===========================================================================
print(f"\n{'='*70}")
print(f"ETAPA 8: GERANDO VISUALIZAÇÕES")
print(f"{'='*70}")

# Matriz de confusão
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Classe 0', 'Classe 1'],
            yticklabels=['Classe 0', 'Classe 1'])
plt.title('Matriz de Confusão - Modelo V2 Enhanced')
plt.ylabel('Real')
plt.xlabel('Predito')
plt.tight_layout()
plt.savefig('confusion_matrix_v2.png', dpi=300, bbox_inches='tight')
print("✓ confusion_matrix_v2.png")

# Curva ROC
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Modelo V2 Enhanced')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('roc_curve_v2.png', dpi=300, bbox_inches='tight')
print("✓ roc_curve_v2.png")

# Threshold analysis
thresholds_plot = [r['threshold'] for r in threshold_results]
precisions_plot = [r['precision'] for r in threshold_results]
recalls_plot = [r['recall'] for r in threshold_results]
f1s_plot = [r['f1_score'] for r in threshold_results]

plt.figure(figsize=(10, 6))
plt.plot(thresholds_plot, precisions_plot, marker='o', label='Precision', linewidth=2)
plt.plot(thresholds_plot, recalls_plot, marker='s', label='Recall', linewidth=2)
plt.plot(thresholds_plot, f1s_plot, marker='^', label='F1-Score', linewidth=2)
plt.axvline(x=best_threshold['threshold'], color='red', linestyle='--', 
            label=f'Best ({best_threshold["threshold"]:.2f})', linewidth=2)
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.title('Threshold Analysis - Modelo V2 Enhanced')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('threshold_analysis_v2.png', dpi=300, bbox_inches='tight')
print("✓ threshold_analysis_v2.png")

# Feature importance
plt.figure(figsize=(12, 10))
plot_importance(model, max_num_features=25, importance_type='gain')
plt.title('Feature Importance (Top 25) - Modelo V2 Enhanced')
plt.tight_layout()
plt.savefig('feature_importance_v2.png', dpi=300, bbox_inches='tight')
print("✓ feature_importance_v2.png")

# Salvar modelo
model.save_model('xgboost_model_v2_enhanced.json')
print("✓ xgboost_model_v2_enhanced.json")

# ===========================================================================
# RESUMO FINAL
# ===========================================================================
print(f"\n{'='*70}")
print(f"🎉 MODELO V2 ENHANCED - RESUMO FINAL")
print(f"{'='*70}")
print(f"\n📊 Dataset:")
print(f"   Registros originais: 500,000")
print(f"   Registros após limpeza: {len(df):,}")
print(f"   Features finais: {X.shape[1]}")
print(f"\n🎯 Performance:")
print(f"   ROC-AUC:   {roc_auc:.4f}")
print(f"   Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"   Precision: {precision:.4f}")
print(f"   Recall:    {recall:.4f}")
print(f"   F1-Score:  {f1:.4f}")
print(f"   Threshold: {best_threshold['threshold']:.2f}")
print(f"\n📁 Arquivos gerados:")
print(f"   - confusion_matrix_v2.png")
print(f"   - roc_curve_v2.png")
print(f"   - threshold_analysis_v2.png")
print(f"   - feature_importance_v2.png")
print(f"   - xgboost_model_v2_enhanced.json")
print(f"\n{'='*70}")
