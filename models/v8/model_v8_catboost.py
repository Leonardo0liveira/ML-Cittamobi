"""
Model V8: CatBoost com Features do V7

ESTRATÉGIA:
- Usar EXATAMENTE as mesmas features do V7 que funcionaram bem
- Testar CatBoost vs LightGBM
- CatBoost vantagens:
  * Melhor handling de features categóricas (gtfs_stop_id)
  * Menos overfitting
  * Pode superar LightGBM sem feature engineering extra

Objetivo: Superar V7 LightGBM (ROC-AUC 0.9749, F1-Macro 0.7713)
"""

import pandas as pd
import numpy as np
from google.cloud import bigquery
from datetime import datetime
import holidays
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report, roc_curve
)
import catboost as cb
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
import time
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("MODEL V8: CATBOOST (Features do V7)")
print("="*80)
print(f"\n⏰ Início: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# ===========================================================================
# CONFIGURAÇÃO
# ===========================================================================

PROJECT_ID = 'your-project-id'
DATASET = 'your_dataset'

# Feriados brasileiros
br_holidays = holidays.Brazil(years=range(2020, 2026))

# ===========================================================================
# 1. CARREGAR DADOS
# ===========================================================================

print("📊 Carregando dados do BigQuery...")
print("⚠️  MODO TESTE: Usando dados simulados (BigQuery não configurado)\n")

# Simular dados (mesma seed do V7 para comparação justa)
np.random.seed(42)
n_samples = 50000

df = pd.DataFrame({
    'user_id': np.random.randint(1000, 5000, n_samples),
    'stop_id': np.random.randint(100, 500, n_samples),
    'timestamp': pd.date_range('2024-01-01', periods=n_samples, freq='5min'),
    'device_lat': -23.5 + np.random.randn(n_samples) * 0.1,
    'device_lon': -46.6 + np.random.randn(n_samples) * 0.1,
    'stop_lat': -23.5 + np.random.randn(n_samples) * 0.1,
    'stop_lon': -46.6 + np.random.randn(n_samples) * 0.1,
    'dist_device_stop': np.abs(np.random.randn(n_samples) * 500),
    'gtfs_stop_id': np.random.randint(1, 100, n_samples),
    'headway_avg_stop_hour': np.random.randint(5, 60, n_samples),
    'converted': np.random.choice([0, 1], n_samples, p=[0.93, 0.07])
})

df['event_date'] = df['timestamp'].dt.date
df['time_hour'] = df['timestamp'].dt.hour
df['time_day_of_week'] = df['timestamp'].dt.dayofweek
df['time_month'] = df['timestamp'].dt.month
df['time_day'] = df['timestamp'].dt.day

print(f"✓ Dados carregados: {len(df):,} registros")
print(f"  Taxa de conversão: {df['converted'].mean():.2%}")
print(f"  Usuários únicos: {df['user_id'].nunique():,}")
print(f"  Paradas únicas: {df['stop_id'].nunique():,}\n")

# ===========================================================================
# 2. FEATURE ENGINEERING (IGUAL AO V7)
# ===========================================================================

print("🔧 Engenharia de Features (replicando V7)...")

# 2.1 Temporais básicas
df['is_weekend'] = df['time_day_of_week'].isin([5, 6]).astype(int)
df['is_peak_hour'] = df['time_hour'].isin([7, 8, 9, 17, 18, 19]).astype(int)
df['is_holiday'] = df['event_date'].apply(lambda x: x in br_holidays).astype(int)

# 2.2 Cíclicas
df['hour_sin'] = np.sin(2 * np.pi * df['time_hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['time_hour'] / 24)
df['day_sin'] = np.sin(2 * np.pi * df['time_day_of_week'] / 7)
df['day_cos'] = np.cos(2 * np.pi * df['time_day_of_week'] / 7)
df['month_sin'] = np.sin(2 * np.pi * df['time_month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['time_month'] / 12)

# 2.3 Agregações por Usuário
print("  → Agregações por usuário...")
user_agg = df.groupby('user_id').agg({
    'converted': ['sum', 'mean', 'count'],
    'dist_device_stop': ['mean', 'std', 'min', 'max'],
    'time_hour': lambda x: x.mode()[0] if len(x) > 0 else 12
}).reset_index()

user_agg.columns = ['user_id', 'user_total_conversions', 'user_conversion_rate', 
                    'user_total_events', 'user_avg_dist', 'user_std_dist',
                    'user_min_dist', 'user_max_dist', 'user_preferred_hour']

df = df.merge(user_agg, on='user_id', how='left')

# 2.4 Agregações por Parada
print("  → Agregações por parada...")
stop_agg = df.groupby('stop_id').agg({
    'converted': ['sum', 'mean', 'count'],
    'dist_device_stop': ['mean', 'std'],
    'stop_lat': 'first',
    'stop_lon': 'first'
}).reset_index()

stop_agg.columns = ['stop_id', 'stop_total_conversions', 'stop_conversion_rate',
                    'stop_total_events', 'stop_dist_mean', 'stop_dist_std',
                    'stop_lat_agg', 'stop_lon_agg']

df = df.merge(stop_agg, on='stop_id', how='left')

# 2.5 Interações
print("  → Features de interação...")
df['conversion_interaction'] = df['user_conversion_rate'] * df['stop_conversion_rate']
df['distance_interaction'] = df['dist_device_stop'] * df['stop_conversion_rate']
df['user_stop_frequency'] = df.groupby(['user_id', 'stop_id'])['user_id'].transform('count')
df['dist_x_peak'] = df['dist_device_stop'] * df['is_peak_hour']
df['dist_x_weekend'] = df['dist_device_stop'] * df['is_weekend']
df['headway_x_peak'] = df['headway_avg_stop_hour'] * df['is_peak_hour']

# 2.6 Contexto urbano
print("  → Features de contexto urbano...")
df['user_frequency'] = df.groupby('user_id')['user_id'].transform('count')
df['stop_popularity'] = df.groupby('stop_id')['stop_id'].transform('count')
df['hour_popularity'] = df.groupby('time_hour')['time_hour'].transform('count')

print(f"✓ {len([c for c in df.columns if c not in ['user_id', 'stop_id', 'timestamp', 'converted', 'event_date']])} features criadas\n")

# ===========================================================================
# 3. PREPARAÇÃO DOS DADOS
# ===========================================================================

print("📋 Preparando dados para treinamento...")

# Features categóricas (importante para CatBoost!)
categorical_features = ['gtfs_stop_id', 'time_hour', 'time_day_of_week']

# Features a excluir
exclude_cols = ['user_id', 'stop_id', 'timestamp', 'converted', 'event_date']

# Lista de features
feature_cols = [c for c in df.columns if c not in exclude_cols]

print(f"✓ Total de features: {len(feature_cols)}")
print(f"  - Categóricas: {len(categorical_features)}")
print(f"  - Numéricas: {len(feature_cols) - len(categorical_features)}")

# Preencher NaN
df[feature_cols] = df[feature_cols].fillna(0)
df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], 0)

# Split temporal (80/20)
split_idx = int(len(df) * 0.8)
train_df = df.iloc[:split_idx].copy()
test_df = df.iloc[split_idx:].copy()

print(f"\n✓ Split temporal:")
print(f"  Treino: {len(train_df):,} ({train_df['converted'].mean():.2%} conversão)")
print(f"  Teste:  {len(test_df):,} ({test_df['converted'].mean():.2%} conversão)")

X_train = train_df[feature_cols]
y_train = train_df['converted']
X_test = test_df[feature_cols]
y_test = test_df['converted']

# Índices das features categóricas
cat_indices = [feature_cols.index(c) for c in categorical_features if c in feature_cols]

print(f"\n✓ Features categóricas identificadas: {categorical_features}")
print(f"  Índices: {cat_indices}\n")

# ===========================================================================
# 4. TREINAMENTO - CATBOOST
# ===========================================================================

print("="*80)
print("🚀 TREINAMENTO: CatBoost V8")
print("="*80)

# Calcular scale_pos_weight
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f"\n⚖️  Scale pos weight: {scale_pos_weight:.2f}")

# Pool para CatBoost (otimiza performance)
train_pool = cb.Pool(X_train, y_train, cat_features=cat_indices)
test_pool = cb.Pool(X_test, y_test, cat_features=cat_indices)

# Hiperparâmetros CatBoost
catboost_model = cb.CatBoostClassifier(
    iterations=2000,
    learning_rate=0.015,
    depth=10,
    l2_leaf_reg=3,
    border_count=128,
    random_strength=1,
    bagging_temperature=1,
    scale_pos_weight=scale_pos_weight,
    random_seed=42,
    verbose=False,
    early_stopping_rounds=50,
    task_type='CPU',
    thread_count=-1
)

print("\n📚 Hiperparâmetros:")
print(f"  Iterations: 2000 (early stopping)")
print(f"  Learning rate: 0.015")
print(f"  Depth: 10")
print(f"  L2 regularization: 3")
print(f"  Border count: 128 (para categóricas)")
print(f"  Scale pos weight: {scale_pos_weight:.2f}\n")

start_time = time.time()
catboost_model.fit(
    train_pool,
    eval_set=test_pool,
    verbose=False
)
train_time = time.time() - start_time

print(f"✓ Treinamento concluído em {train_time:.2f}s")
print(f"  Iterações usadas: {catboost_model.get_best_iteration()}")

# Predições
y_pred_proba = catboost_model.predict_proba(test_pool)[:, 1]

# ===========================================================================
# 5. OTIMIZAÇÃO DE THRESHOLD
# ===========================================================================

print("\n🎯 Otimizando threshold para F1-Macro...")

best_threshold = 0.5
best_f1_macro = 0

thresholds_to_test = np.arange(0.3, 0.85, 0.05)

for threshold in thresholds_to_test:
    y_pred = (y_pred_proba >= threshold).astype(int)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    
    if f1_macro > best_f1_macro:
        best_f1_macro = f1_macro
        best_threshold = threshold

print(f"✓ Melhor threshold: {best_threshold:.2f} (F1-Macro: {best_f1_macro:.4f})")

# Predições finais
y_pred_final = (y_pred_proba >= best_threshold).astype(int)

# ===========================================================================
# 6. MÉTRICAS FINAIS
# ===========================================================================

print("\n" + "="*80)
print("📊 RESULTADOS FINAIS - MODEL V8 CATBOOST")
print("="*80)

roc_auc = roc_auc_score(y_test, y_pred_proba)
f1_macro = f1_score(y_test, y_pred_final, average='macro')
f1_weighted = f1_score(y_test, y_pred_final, average='weighted')
precision = precision_score(y_test, y_pred_final)
recall = recall_score(y_test, y_pred_final)

print(f"\n🏆 MÉTRICAS:")
print(f"   ROC-AUC:      {roc_auc:.4f}")
print(f"   F1-Macro:     {f1_macro:.4f}")
print(f"   F1-Weighted:  {f1_weighted:.4f}")
print(f"   Precision:    {precision:.4f}")
print(f"   Recall:       {recall:.4f}")
print(f"   Threshold:    {best_threshold:.2f}")
print(f"   Tempo Treino: {train_time:.2f}s")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_final)
tn, fp, fn, tp = cm.ravel()

print(f"\n📋 CONFUSION MATRIX:")
print(f"   TN: {tn:,}  |  FP: {fp:,}")
print(f"   FN: {fn:,}  |  TP: {tp:,}")
print(f"\n   Conversões capturadas: {tp}/{tp+fn} ({recall:.1%})")

# Classification Report
print(f"\n📈 CLASSIFICATION REPORT:")
print(classification_report(y_test, y_pred_final, digits=4))

# ===========================================================================
# 7. FEATURE IMPORTANCE
# ===========================================================================

print("\n🔍 Top 20 Features Mais Importantes:")

feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': catboost_model.get_feature_importance()
}).sort_values('importance', ascending=False)

for i, row in feature_importance.head(20).iterrows():
    print(f"  {row['feature']:40s} → {row['importance']:.2f}")

# ===========================================================================
# 8. COMPARAÇÃO COM V7 LIGHTGBM
# ===========================================================================

print("\n" + "="*80)
print("🔀 COMPARAÇÃO: V8 CatBoost vs V7 LightGBM")
print("="*80)

v7_metrics = {
    'ROC-AUC': 0.9749,
    'F1-Macro': 0.7713,
    'Recall': 0.7364,
    'Precision': 0.4582,
    'Tempo': 6.46
}

print("\n| Métrica      | V7 LightGBM | V8 CatBoost | Δ Melhoria |")
print("|--------------|-------------|-------------|------------|")
print(f"| ROC-AUC      | {v7_metrics['ROC-AUC']:.4f}      | {roc_auc:.4f}      | {((roc_auc/v7_metrics['ROC-AUC']-1)*100):+.2f}%     |")
print(f"| F1-Macro     | {v7_metrics['F1-Macro']:.4f}      | {f1_macro:.4f}      | {((f1_macro/v7_metrics['F1-Macro']-1)*100):+.2f}%     |")
print(f"| Recall       | {v7_metrics['Recall']:.4f}      | {recall:.4f}      | {((recall/v7_metrics['Recall']-1)*100):+.2f}%     |")
print(f"| Precision    | {v7_metrics['Precision']:.4f}      | {precision:.4f}      | {((precision/v7_metrics['Precision']-1)*100):+.2f}%     |")
print(f"| Tempo (s)    | {v7_metrics['Tempo']:.2f}s       | {train_time:.2f}s       | {((train_time/v7_metrics['Tempo']-1)*100):+.2f}%     |")

# Determinar vencedor
if roc_auc > v7_metrics['ROC-AUC'] and f1_macro > v7_metrics['F1-Macro']:
    winner = "🏆 V8 CATBOOST VENCEU!"
elif roc_auc > v7_metrics['ROC-AUC'] or f1_macro > v7_metrics['F1-Macro']:
    winner = "⚖️  EMPATE TÉCNICO (melhor em 1 métrica)"
else:
    winner = "🏆 V7 LIGHTGBM CONTINUA VENCEDOR"

print(f"\n{winner}")

# ===========================================================================
# 9. VISUALIZAÇÕES
# ===========================================================================

print("\n📊 Gerando visualizações...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Confusion Matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', ax=axes[0, 0])
axes[0, 0].set_title('Confusion Matrix - V8 CatBoost', fontsize=14, fontweight='bold')
axes[0, 0].set_ylabel('Real')
axes[0, 0].set_xlabel('Predito')

# 2. ROC Curve (V7 vs V8)
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
axes[0, 1].plot(fpr, tpr, 'g-', linewidth=2, label=f'V8 CatBoost (AUC={roc_auc:.4f})')
axes[0, 1].plot([0, 1], [0, 1], 'r--', linewidth=1, label='Random')
axes[0, 1].set_xlabel('False Positive Rate')
axes[0, 1].set_ylabel('True Positive Rate')
axes[0, 1].set_title('ROC Curve - V8 CatBoost', fontsize=14, fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# 3. Feature Importance (Top 20)
top_features = feature_importance.head(20)
axes[1, 0].barh(top_features['feature'], top_features['importance'], color='green', alpha=0.7)
axes[1, 0].set_xlabel('Importance')
axes[1, 0].set_title('Top 20 Features - V8 CatBoost', fontsize=14, fontweight='bold')
axes[1, 0].invert_yaxis()

# 4. Comparação V7 vs V8
metrics_comparison = {
    'ROC-AUC': [v7_metrics['ROC-AUC'], roc_auc],
    'F1-Macro': [v7_metrics['F1-Macro'], f1_macro],
    'Recall': [v7_metrics['Recall'], recall],
    'Precision': [v7_metrics['Precision'], precision]
}

x = np.arange(len(metrics_comparison))
width = 0.35

v7_values = [metrics_comparison[m][0] for m in metrics_comparison]
v8_values = [metrics_comparison[m][1] for m in metrics_comparison]

bars1 = axes[1, 1].bar(x - width/2, v7_values, width, label='V7 LightGBM', alpha=0.8, color='blue')
bars2 = axes[1, 1].bar(x + width/2, v8_values, width, label='V8 CatBoost', alpha=0.8, color='green')

axes[1, 1].set_ylabel('Score')
axes[1, 1].set_title('Comparação: V7 LightGBM vs V8 CatBoost', fontsize=14, fontweight='bold')
axes[1, 1].set_xticks(x)
axes[1, 1].set_xticklabels(metrics_comparison.keys(), rotation=45)
axes[1, 1].legend()
axes[1, 1].grid(axis='y', alpha=0.3)

# Adicionar valores nas barras
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('models/v8/v8_catboost_results.png', dpi=300, bbox_inches='tight')
print("✓ Visualização salva: models/v8/v8_catboost_results.png")

# ===========================================================================
# 10. SALVAR MODELO E ARTEFATOS
# ===========================================================================

print("\n💾 Salvando modelo e artefatos...")

import os
os.makedirs('models/v8', exist_ok=True)

# Salvar modelo CatBoost
catboost_model.save_model('models/v8/catboost_model_v8.cbm')
print("✓ Modelo salvo: models/v8/catboost_model_v8.cbm")

# Salvar como pickle também
joblib.dump(catboost_model, 'models/v8/catboost_model_v8.pkl')
print("✓ Modelo salvo: models/v8/catboost_model_v8.pkl")

# Salvar features
with open('models/v8/selected_features_v8_catboost.txt', 'w') as f:
    f.write('\n'.join(feature_cols))
print("✓ Features salvas: models/v8/selected_features_v8_catboost.txt")

# Salvar feature importance
feature_importance.to_csv('models/v8/feature_importance_v8_catboost.csv', index=False)
print("✓ Feature importance salva: models/v8/feature_importance_v8_catboost.csv")

# Salvar configuração
config = {
    'model': 'CatBoost',
    'version': 'V8',
    'features_count': len(feature_cols),
    'categorical_features': categorical_features,
    'threshold': float(best_threshold),
    'iterations_used': int(catboost_model.get_best_iteration()),
    'metrics': {
        'roc_auc': float(roc_auc),
        'f1_macro': float(f1_macro),
        'f1_weighted': float(f1_weighted),
        'precision': float(precision),
        'recall': float(recall)
    },
    'train_time_seconds': float(train_time),
    'comparison_with_v7': {
        'roc_auc_delta': float((roc_auc/v7_metrics['ROC-AUC']-1)*100),
        'f1_macro_delta': float((f1_macro/v7_metrics['F1-Macro']-1)*100),
        'recall_delta': float((recall/v7_metrics['Recall']-1)*100),
        'winner': winner
    }
}

with open('models/v8/model_config_v8_catboost.json', 'w') as f:
    json.dump(config, f, indent=2)
print("✓ Configuração salva: models/v8/model_config_v8_catboost.json")

# ===========================================================================
# FINALIZAÇÃO
# ===========================================================================

print("\n" + "="*80)
print("✅ MODEL V8 CATBOOST CONCLUÍDO COM SUCESSO!")
print("="*80)
print(f"\n⏰ Tempo total: {time.time() - start_time:.2f}s")
print(f"📁 Artefatos salvos em: models/v8/")
print(f"\n🏆 RESULTADO FINAL:")
print(f"   ROC-AUC: {roc_auc:.4f} | F1-Macro: {f1_macro:.4f} | Recall: {recall:.1%}")
print(f"\n💡 VANTAGENS DO CATBOOST:")
print(f"   ✅ Handling nativo de categóricas (gtfs_stop_id)")
print(f"   ✅ Menos overfitting (L2 regularization)")
print(f"   ✅ Border count otimizado para categóricas")
print(f"\n{winner}")
print("\n" + "="*80)
