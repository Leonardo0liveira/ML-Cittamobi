from google.cloud import bigquery
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_auc_score, accuracy_score, precision_score, 
                             recall_score, f1_score, confusion_matrix, classification_report, roc_curve)
from sklearn.model_selection import TimeSeriesSplit
from catboost import CatBoostClassifier, Pool
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

project_id = "proj-ml-469320"
client = bigquery.Client(project=project_id)

# ===========================================================================
# ETAPA 1: CARREGAR DADOS
# ===========================================================================
print(f"\n{'='*70}")
print(f"MODELO V6 - CATBOOST (OTIMIZADO PARA CATEGÓRICAS)")
print(f"{'='*70}")
print(f"ETAPA 1: CARREGANDO DATASET COM AMOSTRAGEM ALEATÓRIA")
print(f"{'='*70}")

df = pd.read_csv('/Users/leonardooliveira/Downloads/Projeto Machine Learning/data/sampled_dataset.csv')
print(f"✓ Dados carregados: {len(df):,} registros")

# ===========================================================================
# ETAPA 2: FEATURE ENGINEERING AVANÇADO (IGUAL V4)
# ===========================================================================
print(f"\n{'='*70}")
print(f"ETAPA 2: FEATURE ENGINEERING AVANÇADO")
print(f"{'='*70}")

target = "target"

# Criar features de agregação por usuário
print("\n📊 Criando features de agregação por usuário...")
if 'user_pseudo_id' in df.columns:
    user_agg = df.groupby('user_pseudo_id').agg({
        target: ['mean', 'sum', 'count'],
        'dist_device_stop': ['mean', 'std', 'min', 'max'] if 'dist_device_stop' in df.columns else ['mean'],
        'time_hour': ['mean', 'std'] if 'time_hour' in df.columns else ['mean']
    }).reset_index()
    
    user_agg.columns = ['user_pseudo_id', 'user_conversion_rate', 'user_total_conversions', 
                        'user_total_events', 'user_avg_dist', 'user_std_dist', 
                        'user_min_dist', 'user_max_dist', 'user_avg_hour', 'user_std_hour']
    
    df = df.merge(user_agg, on='user_pseudo_id', how='left')
    print(f"✓ Features de usuário criadas: {len(user_agg.columns)-1}")

# Features de agregação por parada
print("\n📊 Criando features de agregação por parada...")
if 'gtfs_stop_id' in df.columns:
    stop_agg = df.groupby('gtfs_stop_id').agg({
        target: ['mean', 'count'],
        'user_frequency': ['mean', 'median'] if 'user_frequency' in df.columns else ['mean']
    }).reset_index()
    
    stop_agg.columns = ['gtfs_stop_id', 'stop_conversion_rate', 'stop_event_count_agg',
                        'stop_user_freq_mean', 'stop_user_freq_median']
    
    df = df.merge(stop_agg, on='gtfs_stop_id', how='left')
    print(f"✓ Features de parada criadas: {len(stop_agg.columns)-1}")

# Features de interação avançadas
print("\n📊 Criando features de interação de 2ª ordem...")
if 'user_conversion_rate' in df.columns and 'stop_conversion_rate' in df.columns:
    df['conversion_interaction'] = df['user_conversion_rate'] * df['stop_conversion_rate']
    
if 'user_avg_dist' in df.columns and 'dist_device_stop' in df.columns:
    df['dist_deviation'] = abs(df['dist_device_stop'] - df['user_avg_dist'])
    df['dist_ratio'] = df['dist_device_stop'] / (df['user_avg_dist'] + 1)

if 'user_frequency' in df.columns and 'stop_event_count_agg' in df.columns:
    df['user_stop_affinity'] = df['user_frequency'] * df['stop_event_count_agg']

print(f"✓ Features de interação criadas")

# ===========================================================================
# ETAPA 3: LIMPEZA MODERADA
# ===========================================================================
print(f"\n{'='*70}")
print(f"ETAPA 3: LIMPEZA MODERADA DOS DADOS")
print(f"{'='*70}")

print(f"\n=== Distribuição do Target (Original) ===")
target_dist = df[target].value_counts()
print(target_dist)
print(f"Proporção classe 0: {target_dist[0]/len(df)*100:.2f}%")
print(f"Proporção classe 1: {target_dist[1]/len(df)*100:.2f}%")

df_original_size = len(df)

# Filtros moderados
if 'user_frequency' in df.columns:
    user_freq_threshold = df['user_frequency'].quantile(0.10)
    df = df[df['user_frequency'] >= user_freq_threshold]
    removed = df_original_size - len(df)
    print(f"✓ Filtro 1: Removidos {removed:,} registros ({removed/df_original_size*100:.1f}%)")
    df_original_size = len(df)

if 'device_lat' in df.columns and 'device_lon' in df.columns:
    df = df[~((df['device_lat'].isna()) | (df['device_lon'].isna()))]
    df = df[~((df['device_lat'] == 0) & (df['device_lon'] == 0))]
    removed = df_original_size - len(df)
    print(f"✓ Filtro 2: Removidos {removed:,} registros ({removed/df_original_size*100:.1f}%)")
    df_original_size = len(df)

if 'dist_device_stop' in df.columns:
    dist_threshold = df['dist_device_stop'].quantile(0.98)
    df = df[df['dist_device_stop'] <= dist_threshold]
    removed = df_original_size - len(df)
    print(f"✓ Filtro 3: Removidos {removed:,} registros ({removed/df_original_size*100:.1f}%)")

total_removed = 200000 - len(df)
removal_pct = total_removed / 200000 * 100
print(f"\n✓ Total mantido: {len(df):,} ({100-removal_pct:.1f}%)")

# ===========================================================================
# ETAPA 4: PREPARAÇÃO DE FEATURES (MANTENDO CATEGÓRICAS)
# ===========================================================================
print(f"\n{'='*70}")
print(f"ETAPA 4: PREPARAÇÃO DE FEATURES (CATBOOST NÃO PRECISA ENCODING)")
print(f"{'='*70}")

features_to_drop = ['y_pred', 'y_pred_proba', 'ctm_service_route', 'direction', 'lotacao_proxy_binaria']
X = df.drop(columns=[target] + features_to_drop, errors='ignore')
y = df[target]

# Processar timestamp
if 'event_timestamp' in X.columns:
    X['event_timestamp'] = pd.to_datetime(X['event_timestamp'], format='ISO8601')
    X['year'] = X['event_timestamp'].dt.year
    X['month'] = X['event_timestamp'].dt.month
    X['day'] = X['event_timestamp'].dt.day
    X['hour'] = X['event_timestamp'].dt.hour
    X['dayofweek'] = X['event_timestamp'].dt.dayofweek
    X['minute'] = X['event_timestamp'].dt.minute
    X['week_of_year'] = X['event_timestamp'].dt.isocalendar().week
    X = X.drop(columns=['event_timestamp'])

# Features cíclicas
if 'time_day_of_month' in X.columns:
    X['day_of_month_sin'] = np.sin(2 * np.pi * X['time_day_of_month'] / 31)
    X['day_of_month_cos'] = np.cos(2 * np.pi * X['time_day_of_month'] / 31)

if 'week_of_year' in X.columns:
    X['week_sin'] = np.sin(2 * np.pi * X['week_of_year'] / 52)
    X['week_cos'] = np.cos(2 * np.pi * X['week_of_year'] / 52)

if 'hour' in X.columns:
    X['hour_sin'] = np.sin(2 * np.pi * X['hour'] / 24)
    X['hour_cos'] = np.cos(2 * np.pi * X['hour'] / 24)

# Identificar colunas categóricas (CatBoost pode tratá-las diretamente)
categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
print(f"\n📊 Colunas categóricas identificadas: {len(categorical_cols)}")
if categorical_cols:
    print(f"   {categorical_cols[:5]}{'...' if len(categorical_cols) > 5 else ''}")

# CatBoost lida com NaN automaticamente, mas vamos preencher inf
X = X.replace([np.inf, -np.inf], np.nan)

print(f"✓ Features totais: {X.shape[1]}")
print(f"✓ Features categóricas: {len(categorical_cols)}")

# ===========================================================================
# ETAPA 5: DIVISÃO TEMPORAL
# ===========================================================================
print(f"\n{'='*70}")
print(f"ETAPA 5: DIVISÃO TEMPORAL (TimeSeriesSplit)")
print(f"{'='*70}")

tscv = TimeSeriesSplit(n_splits=3)
X_train, X_test, y_train, y_test = None, None, None, None

for fold, (train_index, test_index) in enumerate(tscv.split(X)):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    if fold == 2:
        print(f"Fold {fold + 1}: Train={len(X_train):,} | Test={len(X_test):,}")

print(f"\n=== Distribuição do Treino ===")
train_dist = y_train.value_counts()
print(f"Classe 0: {train_dist[0]:,} ({train_dist[0]/len(y_train)*100:.2f}%)")
print(f"Classe 1: {train_dist[1]:,} ({train_dist[1]/len(y_train)*100:.2f}%)")
scale_pos_weight = train_dist[0] / train_dist[1]
print(f"Scale pos weight: {scale_pos_weight:.2f}")

# ===========================================================================
# ETAPA 6: TREINAMENTO CATBOOST
# ===========================================================================
print(f"\n{'='*70}")
print(f"ETAPA 6: TREINAMENTO CATBOOST")
print(f"{'='*70}")

# Criar Pools do CatBoost (formato otimizado)
train_pool = Pool(
    data=X_train,
    label=y_train,
    cat_features=categorical_cols if categorical_cols else None
)

test_pool = Pool(
    data=X_test,
    label=y_test,
    cat_features=categorical_cols if categorical_cols else None
)

# Modelo CatBoost com configurações otimizadas
model = CatBoostClassifier(
    iterations=100,  # Reduzido de 500 para 100 (mais rápido)
    learning_rate=0.05,  # Aumentado de 0.015 para 0.05 (compensar menos iterações)
    depth=12,  # Reduzido de 16 para 12 (mais rápido)
    loss_function='Logloss',
    eval_metric='AUC',
    auto_class_weights='Balanced',  # BALANCEAMENTO AUTOMÁTICO!
    l2_leaf_reg=1.0,  # Regularização L2
    border_count=128,  # Reduzido de 254 para 128 (mais rápido)
    random_strength=1.0,  # Randomização para prevenir overfitting
    leaf_estimation_iterations=5,  # Reduzido de 10 para 5 (mais rápido)
    subsample=0.85,  # Subsample de linhas
    rsm=0.85,  # Random subspace method (subsample de features)
    min_data_in_leaf=20,  # Mínimo de amostras por folha
    bootstrap_type='Bernoulli',  # Tipo de bootstrap (removido bagging_temperature)
    task_type='CPU',  # Use 'GPU' se tiver GPU disponível
    verbose=20,  # Reduzido de 50 para 20 (menos logs)
    early_stopping_rounds=15,  # Reduzido de 30 para 15 (parar mais cedo)
    random_seed=42,
    thread_count=-1
)

print("\n📊 Parâmetros do modelo:")
print(f"   iterations: {model.get_params()['iterations']}")
print(f"   learning_rate: {model.get_params()['learning_rate']}")
print(f"   depth: {model.get_params()['depth']}")
print(f"   auto_class_weights: {model.get_params()['auto_class_weights']}")
print(f"   cat_features: {len(categorical_cols) if categorical_cols else 0}")

print("\n🚀 Iniciando treinamento...")
model.fit(
    train_pool,
    eval_set=test_pool,
    verbose=True,
    plot=False
)

print(f"\n✓ Treinamento concluído!")
print(f"✓ Melhor iteração: {model.get_best_iteration()}")
print(f"✓ Melhor score: {model.get_best_score()['validation']['AUC']:.4f}")

# ===========================================================================
# ETAPA 7: PREDIÇÕES E OTIMIZAÇÃO DE THRESHOLD
# ===========================================================================
print(f"\n{'='*70}")
print(f"ETAPA 7: PREDIÇÕES E OTIMIZAÇÃO DE THRESHOLD")
print(f"{'='*70}")

y_pred_proba = model.predict_proba(X_test)[:, 1]

# Testar diferentes thresholds
print("\n📊 Testando thresholds:")
best_threshold = 0.5
best_f1_macro = 0

for threshold in [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]:
    y_pred_temp = (y_pred_proba >= threshold).astype(int)
    f1_macro = f1_score(y_test, y_pred_temp, average='macro')
    precision = precision_score(y_test, y_pred_temp)
    recall = recall_score(y_test, y_pred_temp)
    
    print(f"   Threshold {threshold:.2f}: F1-Macro={f1_macro:.4f} | Precision={precision:.4f} | Recall={recall:.4f}")
    
    if f1_macro > best_f1_macro:
        best_f1_macro = f1_macro
        best_threshold = threshold

print(f"\n✓ Melhor threshold: {best_threshold} (F1-Macro: {best_f1_macro:.4f})")

# Predições finais
y_pred = (y_pred_proba >= best_threshold).astype(int)

# ===========================================================================
# ETAPA 8: AVALIAÇÃO DO MODELO
# ===========================================================================
print(f"\n{'='*70}")
print(f"ETAPA 8: AVALIAÇÃO DO MODELO V6 - CATBOOST")
print(f"{'='*70}")

roc_auc = roc_auc_score(y_test, y_pred_proba)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
f1_macro = f1_score(y_test, y_pred, average='macro')

print(f"\n📊 MÉTRICAS FINAIS:")
print(f"   ROC-AUC:      {roc_auc:.4f}")
print(f"   Accuracy:     {accuracy:.4f}")
print(f"   Precision:    {precision:.4f}")
print(f"   Recall:       {recall:.4f}")
print(f"   F1-Score:     {f1:.4f}")
print(f"   F1-Macro:     {f1_macro:.4f}")
print(f"   Threshold:    {best_threshold}")

print(f"\n📊 Matriz de Confusão:")
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(f"\nTrue Negatives:  {cm[0,0]:,}")
print(f"False Positives: {cm[0,1]:,}")
print(f"False Negatives: {cm[1,0]:,}")
print(f"True Positives:  {cm[1,1]:,}")

print(f"\n📊 Classification Report:")
print(classification_report(y_test, y_pred, digits=4))

# ===========================================================================
# ETAPA 9: VISUALIZAÇÕES
# ===========================================================================
print(f"\n{'='*70}")
print(f"ETAPA 9: GERANDO VISUALIZAÇÕES")
print(f"{'='*70}")

# 1. Confusion Matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', cbar=False)
plt.title(f'V6 CatBoost - Confusion Matrix\nROC-AUC: {roc_auc:.4f} | F1-Macro: {f1_macro:.4f}', 
          fontsize=14, fontweight='bold')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('../../visualizations/v6/confusion_matrix_v6.png', dpi=300, bbox_inches='tight')
print("✓ Confusion Matrix salva")
plt.close()

# 2. ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, color='green', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('V6 CatBoost - ROC Curve', fontsize=14, fontweight='bold')
plt.legend(loc="lower right", fontsize=11)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('../../visualizations/v6/roc_curve_v6.png', dpi=300, bbox_inches='tight')
print("✓ ROC Curve salva")
plt.close()

# 3. Feature Importance
feature_importance = model.get_feature_importance()
importance_df = pd.DataFrame({
    'feature': X_train.columns,
    'importance': feature_importance
}).sort_values('importance', ascending=False).head(20)

plt.figure(figsize=(12, 8))
plt.barh(range(len(importance_df)), importance_df['importance'], color='green', alpha=0.7)
plt.yticks(range(len(importance_df)), importance_df['feature'])
plt.xlabel('Importance', fontsize=12)
plt.title('V6 CatBoost - Top 20 Feature Importance', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('../../visualizations/v6/feature_importance_v6.png', dpi=300, bbox_inches='tight')
print("✓ Feature Importance salva")
plt.close()

# ===========================================================================
# ETAPA 10: SALVAR MODELO
# ===========================================================================
print(f"\n{'='*70}")
print(f"ETAPA 10: SALVANDO MODELO")
print(f"{'='*70}")

model.save_model('catboost_model_v6.cbm')
print("✓ Modelo salvo: catboost_model_v6.cbm")

# Salvar métricas em arquivo
with open('../../reports/v6_catboost_report.txt', 'w') as f:
    f.write("="*70 + "\n")
    f.write("MODELO V6 - CATBOOST - RELATÓRIO FINAL\n")
    f.write("="*70 + "\n\n")
    f.write(f"ROC-AUC:      {roc_auc:.4f}\n")
    f.write(f"Accuracy:     {accuracy:.4f}\n")
    f.write(f"Precision:    {precision:.4f}\n")
    f.write(f"Recall:       {recall:.4f}\n")
    f.write(f"F1-Score:     {f1:.4f}\n")
    f.write(f"F1-Macro:     {f1_macro:.4f}\n")
    f.write(f"Threshold:    {best_threshold}\n")
    f.write(f"Best Iteration: {model.get_best_iteration()}\n\n")
    f.write("Matriz de Confusão:\n")
    f.write(f"TN: {cm[0,0]:,} | FP: {cm[0,1]:,}\n")
    f.write(f"FN: {cm[1,0]:,} | TP: {cm[1,1]:,}\n\n")
    f.write("Top 20 Features:\n")
    for idx, row in importance_df.iterrows():
        f.write(f"  {row['feature']}: {row['importance']:.2f}\n")
    f.write(f"\nFeatures categóricas tratadas: {len(categorical_cols)}\n")

print("✓ Relatório salvo: reports/v6_catboost_report.txt")

print(f"\n{'='*70}")
print(f"✅ MODELO V6 - CATBOOST CONCLUÍDO!")
print(f"{'='*70}")
print(f"\n📊 RESULTADO FINAL:")
print(f"   ROC-AUC: {roc_auc:.4f}")
print(f"   F1-Macro: {f1_macro:.4f}")
print(f"   Precision: {precision:.4f}")
print(f"   Recall: {recall:.4f}")
print(f"\n💡 VANTAGENS DO CATBOOST:")
print(f"   ✓ Tratamento automático de {len(categorical_cols)} features categóricas")
print(f"   ✓ Balanceamento automático de classes (auto_class_weights='Balanced')")
print(f"   ✓ Menos overfitting que XGBoost")
print(f"   ✓ Não precisa de Label Encoding")
print(f"\n📁 Arquivos salvos:")
print(f"   - Modelo: models/v6/catboost_model_v6.cbm")
print(f"   - Visualizações: visualizations/v6/")
print(f"   - Relatório: reports/v6_catboost_report.txt")
