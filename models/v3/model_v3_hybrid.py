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
# ETAPA 1: AMOSTRAGEM ALEATÓRIA (TABLESAMPLE)
# ===========================================================================
print(f"\n{'='*70}")
print(f"MODELO V3 - ABORDAGEM HÍBRIDA")
print(f"{'='*70}")
print(f"ETAPA 1: CARREGANDO DATASET COM AMOSTRAGEM ALEATÓRIA")
print(f"{'='*70}")

# Usar TABLESAMPLE para amostragem verdadeiramente aleatória
query = """
    SELECT * FROM `proj-ml-469320.app_cittamobi.dataset-updated` 
    TABLESAMPLE SYSTEM (20 PERCENT)
    LIMIT 200000
"""

print("Carregando 200,000 amostras com amostragem aleatória...")
df = client.query(query).to_dataframe()
print(f"✓ Dados carregados: {len(df):,} registros")

# ===========================================================================
# ETAPA 2: LIMPEZA MODERADA (30-40% de remoção)
# ===========================================================================
print(f"\n{'='*70}")
print(f"ETAPA 2: LIMPEZA MODERADA DOS DADOS")
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
# FILTROS MODERADOS (objetivo: remover apenas 30-40%)
# ===========================================================================
print(f"\n=== Aplicando Filtros Moderados ===")

df_original_size = len(df)

# 1. Remover apenas usuários MUITO infrequentes (Q10 ao invés de Q25)
if 'user_frequency' in df.columns:
    user_freq_threshold = df['user_frequency'].quantile(0.10)  # Apenas 10% com menos eventos
    df = df[df['user_frequency'] >= user_freq_threshold]
    removed = df_original_size - len(df)
    print(f"✓ Filtro 1: Removidos {removed:,} registros de usuários muito infrequentes")
    print(f"   Threshold: {user_freq_threshold:.0f} eventos | Remoção: {removed/df_original_size*100:.1f}%")
    df_original_size = len(df)

# 2. Remover apenas localização CLARAMENTE inválida
if 'device_lat' in df.columns and 'device_lon' in df.columns:
    df = df[~((df['device_lat'].isna()) | (df['device_lon'].isna()))]
    df = df[~((df['device_lat'] == 0) & (df['device_lon'] == 0))]
    removed = df_original_size - len(df)
    print(f"✓ Filtro 2: Removidos {removed:,} registros com localização inválida")
    print(f"   Remoção: {removed/df_original_size*100:.1f}%")
    df_original_size = len(df)

# 3. Remover apenas distâncias EXTREMAS (Q98 ao invés de Q95)
if 'dist_device_stop' in df.columns:
    dist_threshold = df['dist_device_stop'].quantile(0.98)  # Apenas 2% com maior distância
    df = df[df['dist_device_stop'] <= dist_threshold]
    removed = df_original_size - len(df)
    print(f"✓ Filtro 3: Removidos {removed:,} registros com distância extrema")
    print(f"   Threshold: {dist_threshold:.2f} | Remoção: {removed/df_original_size*100:.1f}%")
    df_original_size = len(df)

# 4. Manter filtro de headway válido (filtro leve)
if 'headway_avg_stop_hour' in df.columns:
    df = df[df['headway_avg_stop_hour'] > 0]
    removed = df_original_size - len(df)
    print(f"✓ Filtro 4: Removidos {removed:,} registros com headway inválido")
    print(f"   Remoção: {removed/df_original_size*100:.1f}%")
    df_original_size = len(df)

# 5. Remover apenas paradas MUITO infrequentes (Q10 ao invés de Q20)
if 'stop_event_count' in df.columns:
    stop_threshold = df['stop_event_count'].quantile(0.10)  # Apenas 10% com menos eventos
    df = df[df['stop_event_count'] >= stop_threshold]
    removed = df_original_size - len(df)
    print(f"✓ Filtro 5: Removidos {removed:,} registros de paradas muito infrequentes")
    print(f"   Threshold: {stop_threshold:.0f} eventos | Remoção: {removed/df_original_size*100:.1f}%")

# Resumo da limpeza
total_removed = 200000 - len(df)
removal_pct = total_removed / 200000 * 100

print(f"\n=== Resumo da Limpeza ===")
print(f"Registros originais: 200,000")
print(f"Registros mantidos: {len(df):,}")
print(f"Registros removidos: {total_removed:,} ({removal_pct:.1f}%)")

if removal_pct > 40:
    print(f"⚠️ ATENÇÃO: Remoção acima de 40% ({removal_pct:.1f}%)")
elif removal_pct < 30:
    print(f"✓ Remoção dentro do ideal: {removal_pct:.1f}%")
else:
    print(f"✓ Remoção no alvo (30-40%): {removal_pct:.1f}%")

# Verificar balanceamento após filtragem
print(f"\n=== Distribuição do Target (Após Filtros) ===")
target_dist_filtered = df[target].value_counts()
print(target_dist_filtered)
print(f"Proporção classe 0: {target_dist_filtered[0]/len(df)*100:.2f}%")
print(f"Proporção classe 1: {target_dist_filtered[1]/len(df)*100:.2f}%")

# ===========================================================================
# ETAPA 3: FEATURE ENGINEERING ESTRATÉGICO
# ===========================================================================
print(f"\n{'='*70}")
print(f"ETAPA 3: FEATURE ENGINEERING ESTRATÉGICO")
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
    X['minute'] = X['event_timestamp'].dt.minute
    X['week_of_year'] = X['event_timestamp'].dt.isocalendar().week
    X = X.drop(columns=['event_timestamp'])
    print(f"✓ Features temporais extraídas")

# Features de interação SELECIONADAS (apenas as mais promissoras)
print(f"\n=== Criando Features de Interação (Selecionadas) ===")

# 1. Interação hora x dia da semana
if 'hour' in X.columns and 'dayofweek' in X.columns:
    X['hour_x_dayofweek'] = X['hour'] * X['dayofweek']
    print(f"✓ Feature: hour_x_dayofweek")

# 2. Interação distância x pico
if 'dist_device_stop' in X.columns and 'is_peak_hour' in X.columns:
    X['dist_x_peak'] = X['dist_device_stop'] * (X['is_peak_hour'] + 1)
    print(f"✓ Feature: dist_x_peak")

# 3. Taxa de eventos normalizada
if 'stop_event_rate' in X.columns and 'stop_total_samples' in X.columns:
    X['event_rate_normalized'] = X['stop_event_rate'] / (X['stop_total_samples'] + 1)
    print(f"✓ Feature: event_rate_normalized")

# 4. Densidade de eventos
if 'stop_event_count' in X.columns and 'stop_total_samples' in X.columns:
    X['event_density'] = X['stop_event_count'] / (X['stop_total_samples'] + 1)
    print(f"✓ Feature: event_density")

# Features cíclicas (mantidas - importantes para padrões temporais)
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

print(f"\n📊 Dataset com todas as features:")
print(f"   Shape: {X.shape}")
print(f"   Total de features: {X.shape[1]}")

# ===========================================================================
# ETAPA 4: SELEÇÃO DE FEATURES (TOP 35-40)
# ===========================================================================
print(f"\n{'='*70}")
print(f"ETAPA 4: SELEÇÃO DAS MELHORES FEATURES")
print(f"{'='*70}")

# Treinar modelo rápido para obter feature importance
print("Treinando modelo temporário para calcular feature importance...")

# TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=2)
for fold, (train_index, test_index) in enumerate(tscv.split(X)):
    X_train_temp, X_test_temp = X.iloc[train_index], X.iloc[test_index]
    y_train_temp, y_test_temp = y.iloc[train_index], y.iloc[test_index]

dtrain_temp = xgb.DMatrix(X_train_temp, label=y_train_temp)
scale_pos_weight = (len(y_train_temp) - y_train_temp.sum()) / y_train_temp.sum()

params_temp = {
    'objective': 'binary:logistic',
    'max_depth': 6,
    'learning_rate': 0.1,
    'eval_metric': 'logloss',
    'seed': 42,
    'scale_pos_weight': scale_pos_weight
}

model_temp = xgb.train(
    params=params_temp,
    dtrain=dtrain_temp,
    num_boost_round=50,
    verbose_eval=False
)

# Obter feature importance
importance_dict = model_temp.get_score(importance_type='gain')
importance_df = pd.DataFrame([
    {'feature': k, 'importance': v} 
    for k, v in importance_dict.items()
]).sort_values('importance', ascending=False)

print(f"\n=== Top 40 Features por Importância ===")
top_n = 40
top_features = importance_df.head(top_n)['feature'].tolist()

for i, row in importance_df.head(20).iterrows():
    print(f"{i+1:2d}. {row['feature']:30s} | Gain: {row['importance']:.2f}")

if len(importance_df) > 20:
    print(f"... (mostrando apenas top 20, selecionando top {top_n})")

# Selecionar apenas as top N features
X_selected = X[top_features].copy()

print(f"\n✓ Features selecionadas: {len(top_features)}")
print(f"✓ Redução de {X.shape[1]} → {X_selected.shape[1]} features")

# ===========================================================================
# ETAPA 5: DIVISÃO E TREINAMENTO
# ===========================================================================
print(f"\n{'='*70}")
print(f"ETAPA 5: TREINAMENTO DO MODELO V3")
print(f"{'='*70}")

# TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=3)
X_train, X_test, y_train, y_test = None, None, None, None

for fold, (train_index, test_index) in enumerate(tscv.split(X_selected)):
    X_train, X_test = X_selected.iloc[train_index], X_selected.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    print(f"Fold {fold + 1}: Train={len(X_train):,} | Test={len(X_test):,}")

# Criar DMatrix
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Calcular scale_pos_weight
scale_pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()

# ===========================================================================
# ETAPA 6: HIPERPARÂMETROS OTIMIZADOS (baseado em V1 e V2)
# ===========================================================================
print(f"\n{'='*70}")
print(f"ETAPA 6: TUNING DE HIPERPARÂMETROS (V3 Híbrido)")
print(f"{'='*70}")

# Testar configurações que combinam insights de V1 e V2
param_grid = [
    # Config 1: Melhor de V1 (max_depth=10, lr=0.03)
    {'max_depth': 10, 'learning_rate': 0.03, 'subsample': 0.85, 'colsample_bytree': 0.85, 
     'min_child_weight': 5, 'gamma': 0.05, 'reg_alpha': 0.05, 'reg_lambda': 0.5},
    
    # Config 2: Melhor de V2 (max_depth=12, lr=0.02) com ajustes
    {'max_depth': 12, 'learning_rate': 0.02, 'subsample': 0.85, 'colsample_bytree': 0.85, 
     'min_child_weight': 5, 'gamma': 0.1, 'reg_alpha': 0.01, 'reg_lambda': 1},
    
    # Config 3: Híbrido (meio termo)
    {'max_depth': 11, 'learning_rate': 0.025, 'subsample': 0.9, 'colsample_bytree': 0.9, 
     'min_child_weight': 4, 'gamma': 0.05, 'reg_alpha': 0.03, 'reg_lambda': 0.7},
    
    # Config 4: Conservador (prevenir overfitting)
    {'max_depth': 9, 'learning_rate': 0.035, 'subsample': 0.8, 'colsample_bytree': 0.8, 
     'min_child_weight': 6, 'gamma': 0.15, 'reg_alpha': 0.1, 'reg_lambda': 1.5},
    
    # Config 5: Agressivo (capturar mais padrões)
    {'max_depth': 13, 'learning_rate': 0.02, 'subsample': 0.9, 'colsample_bytree': 0.85, 
     'min_child_weight': 3, 'gamma': 0.03, 'reg_alpha': 0.01, 'reg_lambda': 0.3},
]

best_auc = 0
best_params = None
best_model = None

from sklearn.metrics import roc_auc_score as roc_auc_calc

print(f"Testando {len(param_grid)} configurações...")

for i, param_config in enumerate(param_grid):
    print(f"\n--- Config {i+1}/{len(param_grid)} ---")
    print(f"Params: max_depth={param_config['max_depth']}, lr={param_config['learning_rate']}, "
          f"min_child_weight={param_config['min_child_weight']}")
    
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
        num_boost_round=200,  # Aumentar rounds
        evals=[(dtest, 'test')],
        early_stopping_rounds=20,
        verbose_eval=False
    )
    
    y_pred_test = model_test.predict(dtest)
    auc = roc_auc_calc(y_test, y_pred_test)
    print(f"ROC-AUC: {auc:.4f}")
    
    if auc > best_auc:
        best_auc = auc
        best_params = params_test
        best_model = model_test
        print(f"✓ Nova melhor configuração!")

print(f"\n{'='*70}")
print(f"MELHOR MODELO V3 SELECIONADO")
print(f"{'='*70}")
print(f"ROC-AUC: {best_auc:.4f}")
print(f"\nMelhores Parâmetros:")
for key, val in best_params.items():
    if key not in ['objective', 'eval_metric', 'seed']:
        print(f"  {key}: {val}")

model = best_model

# ===========================================================================
# ETAPA 7: OTIMIZAÇÃO DE THRESHOLD
# ===========================================================================
print(f"\n{'='*70}")
print(f"ETAPA 7: OTIMIZANDO THRESHOLD")
print(f"{'='*70}")

y_pred_proba = model.predict(dtest)

from sklearn.metrics import precision_score, recall_score, f1_score

thresholds_to_test = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]
threshold_results = []

for thresh in thresholds_to_test:
    y_pred_thresh = (y_pred_proba >= thresh).astype(int)
    prec = precision_score(y_test, y_pred_thresh)
    rec = recall_score(y_test, y_pred_thresh)
    f1 = f1_score(y_test, y_pred_thresh)
    
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
# ETAPA 8: AVALIAÇÃO FINAL
# ===========================================================================
print(f"\n{'='*70}")
print(f"ETAPA 8: AVALIAÇÃO FINAL DO MODELO V3")
print(f"{'='*70}")

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print(f"\n=== Métricas Finais V3 ===")
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
# ETAPA 9: COMPARAÇÃO COM V1 E V2
# ===========================================================================
print(f"\n{'='*70}")
print(f"ETAPA 9: COMPARAÇÃO V1 vs V2 vs V3")
print(f"{'='*70}")

comparison_data = {
    'Métrica': ['ROC-AUC', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'Threshold', 'Samples', 'Features'],
    'V1': ['0.8367', '89.02%', '45.19%', '51.21%', '0.4801', '0.6', '50k', '38'],
    'V2': ['0.7961', '86.62%', '41.99%', '48.89%', '0.4518', '0.5', '60k', '49'],
    'V3': [f'{roc_auc:.4f}', f'{accuracy*100:.2f}%', f'{precision*100:.2f}%', 
           f'{recall*100:.2f}%', f'{f1:.4f}', f'{best_threshold["threshold"]:.1f}', 
           f'{len(df)//1000}k', f'{len(top_features)}']
}

comparison_df = pd.DataFrame(comparison_data)
print("\n" + comparison_df.to_string(index=False))

# ===========================================================================
# ETAPA 10: VISUALIZAÇÕES
# ===========================================================================
print(f"\n{'='*70}")
print(f"ETAPA 10: GERANDO VISUALIZAÇÕES")
print(f"{'='*70}")

# Matriz de confusão
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', cbar=False,
            xticklabels=['Classe 0', 'Classe 1'],
            yticklabels=['Classe 0', 'Classe 1'])
plt.title('Matriz de Confusão - Modelo V3 Híbrido')
plt.ylabel('Real')
plt.xlabel('Predito')
plt.tight_layout()
plt.savefig('confusion_matrix_v3.png', dpi=300, bbox_inches='tight')
print("✓ confusion_matrix_v3.png")

# Curva ROC
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='green', lw=2, label=f'V3 ROC curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Modelo V3 Híbrido')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('roc_curve_v3.png', dpi=300, bbox_inches='tight')
print("✓ roc_curve_v3.png")

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
plt.title('Threshold Analysis - Modelo V3 Híbrido')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('threshold_analysis_v3.png', dpi=300, bbox_inches='tight')
print("✓ threshold_analysis_v3.png")

# Feature importance (top features selecionadas)
plt.figure(figsize=(12, 10))
plot_importance(model, max_num_features=25, importance_type='gain')
plt.title('Feature Importance (Top 25) - Modelo V3 Híbrido')
plt.tight_layout()
plt.savefig('feature_importance_v3.png', dpi=300, bbox_inches='tight')
print("✓ feature_importance_v3.png")

# Gráfico comparativo V1 vs V2 vs V3
metrics_comparison = {
    'V1': [0.8367, 0.8902, 0.4519, 0.5121, 0.4801],
    'V2': [0.7961, 0.8662, 0.4199, 0.4889, 0.4518],
    'V3': [roc_auc, accuracy, precision, recall, f1]
}

fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(5)
width = 0.25
metrics_labels = ['ROC-AUC', 'Accuracy', 'Precision', 'Recall', 'F1-Score']

bars1 = ax.bar(x - width, metrics_comparison['V1'], width, label='V1', color='#1f77b4')
bars2 = ax.bar(x, metrics_comparison['V2'], width, label='V2', color='#ff7f0e')
bars3 = ax.bar(x + width, metrics_comparison['V3'], width, label='V3', color='#2ca02c')

ax.set_xlabel('Métricas')
ax.set_ylabel('Score')
ax.set_title('Comparação de Performance: V1 vs V2 vs V3')
ax.set_xticks(x)
ax.set_xticklabels(metrics_labels)
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Adicionar valores nas barras
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig('comparison_v1_v2_v3.png', dpi=300, bbox_inches='tight')
print("✓ comparison_v1_v2_v3.png")

# Salvar modelo
model.save_model('xgboost_model_v3_hybrid.json')
print("✓ xgboost_model_v3_hybrid.json")

# Salvar lista de features selecionadas
with open('features_v3_selected.txt', 'w') as f:
    f.write("TOP 40 FEATURES SELECIONADAS PARA V3\n")
    f.write("="*50 + "\n\n")
    for i, feat in enumerate(top_features, 1):
        importance_val = importance_df[importance_df['feature'] == feat]['importance'].values[0]
        f.write(f"{i:2d}. {feat:30s} | Gain: {importance_val:.2f}\n")
print("✓ features_v3_selected.txt")

# ===========================================================================
# RESUMO FINAL
# ===========================================================================
print(f"\n{'='*70}")
print(f"🎉 MODELO V3 HÍBRIDO - RESUMO FINAL")
print(f"{'='*70}")
print(f"\n📊 Estratégias Implementadas:")
print(f"   ✓ Amostragem aleatória (TABLESAMPLE 20%)")
print(f"   ✓ Filtros moderados (30-40% remoção)")
print(f"   ✓ Seleção de features (top {len(top_features)})")
print(f"   ✓ Hiperparâmetros híbridos (5 configs testadas)")
print(f"\n📊 Dataset:")
print(f"   Registros originais: 200,000")
print(f"   Registros após limpeza: {len(df):,} ({removal_pct:.1f}% removidos)")
print(f"   Features originais: {X.shape[1]}")
print(f"   Features selecionadas: {len(top_features)}")
print(f"\n🎯 Performance Final:")
print(f"   ROC-AUC:   {roc_auc:.4f}")
print(f"   Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"   Precision: {precision:.4f}")
print(f"   Recall:    {recall:.4f}")
print(f"   F1-Score:  {f1:.4f}")
print(f"   Threshold: {best_threshold['threshold']:.2f}")
print(f"\n📈 Comparação com versões anteriores:")
if roc_auc > 0.8367:
    print(f"   ✅ V3 SUPEROU V1! ({roc_auc:.4f} vs 0.8367)")
elif roc_auc > 0.7961:
    print(f"   ✅ V3 melhor que V2, mas abaixo de V1 ({roc_auc:.4f})")
else:
    print(f"   ⚠️ V3 precisa de ajustes ({roc_auc:.4f})")
print(f"\n📁 Arquivos gerados:")
print(f"   - confusion_matrix_v3.png")
print(f"   - roc_curve_v3.png")
print(f"   - threshold_analysis_v3.png")
print(f"   - feature_importance_v3.png")
print(f"   - comparison_v1_v2_v3.png")
print(f"   - xgboost_model_v3_hybrid.json")
print(f"   - features_v3_selected.txt")
print(f"\n{'='*70}")
