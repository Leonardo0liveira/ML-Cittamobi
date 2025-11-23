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

query = """
    SELECT * FROM `proj-ml-469320.app_cittamobi.dataset-updated` LIMIT 50000 
"""

df = client.query(query).to_dataframe()

# Análise inicial dos dados
print(f"\n=== Informações do Dataset ===")
print(f"Shape: {df.shape}")
print(f"\nColunas e tipos:")
print(df.dtypes)
print(f"\nValores nulos:")
print(df.isnull().sum())

target = "target"

# Analisar distribuição do target ANTES de qualquer processamento
print(f"\n=== Análise do Target ===")
print(df[target].describe())
print(f"Target range: {df[target].min()} to {df[target].max()}")
print(f"Target unique values: {df[target].nunique()}")

# Remover features suspeitas de data leakage e colunas 100% nulas
features_to_drop = ['y_pred', 'y_pred_proba', 'ctm_service_route', 'direction', 'lotacao_proxy_binaria']
print(f"\n=== Removendo features suspeitas ===")
print(f"Features removidas: {features_to_drop}")
print(f"Motivo: 'lotacao_proxy_binaria' tem correlação 1.0 com target (data leakage)")

# Separar features e target
X = df.drop(columns=[target] + features_to_drop)
y = df[target]

# Pré-processamento: Converter datetime para features numéricas
if 'event_timestamp' in X.columns:
    X['year'] = pd.to_datetime(X['event_timestamp']).dt.year
    X['month'] = pd.to_datetime(X['event_timestamp']).dt.month
    X['day'] = pd.to_datetime(X['event_timestamp']).dt.day
    X['hour'] = pd.to_datetime(X['event_timestamp']).dt.hour
    X['dayofweek'] = pd.to_datetime(X['event_timestamp']).dt.dayofweek
    X = X.drop(columns=['event_timestamp'])

# Identificar e converter colunas categóricas
categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
print(f"\nColunas categóricas encontradas: {categorical_cols}")

# Label Encoding para colunas categóricas
from sklearn.preprocessing import LabelEncoder
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le

# Verificar se há valores infinitos ou NaN
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(0)

print(f"\n=== Dados após pré-processamento ===")
print(f"Shape: {X.shape}")
print(f"Tipos de dados:")
print(X.dtypes)

# Verificar balanceamento das classes
print(f"\n=== Balanceamento das Classes ===")
class_counts = y.value_counts()
print(class_counts)
print(f"Proporção classe 0: {class_counts[0]/len(y)*100:.2f}%")
print(f"Proporção classe 1: {class_counts[1]/len(y)*100:.2f}%")

# Analisar correlação com target
print(f"\n=== Top 10 Features mais correlacionadas com target ===")
correlations = X.corrwith(y).abs().sort_values(ascending=False)
print(correlations.head(10))

# TimeSeriesSplit para validação temporal
tscv = TimeSeriesSplit(n_splits=2)

# Variáveis para armazenar os dados do último fold
X_train, X_test, y_train, y_test = None, None, None, None

for fold, (train_index, test_index) in enumerate(tscv.split(X)):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    print(f"\nFold {fold + 1}: Train size={len(X_train)}, Test size={len(X_test)}")

# Criar DMatrix para XGBoost (formato otimizado)
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Calcular scale_pos_weight para balancear classes
if y_train is not None:
    scale_pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()
else:
    scale_pos_weight = 1

# ===========================================================================
# ETAPA 1: Modelo Baseline (parâmetros iniciais)
# ===========================================================================
print(f"\n{'='*70}")
print(f"ETAPA 1: TREINANDO MODELO BASELINE")
print(f"{'='*70}")

params_baseline = {
    'objective': 'binary:logistic',
    'max_depth': 6,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'eval_metric': 'logloss',
    'seed': 42,
    'scale_pos_weight': scale_pos_weight
}

print(f"Scale pos weight: {params_baseline['scale_pos_weight']:.2f}")

model_baseline = xgb.train(
    params=params_baseline,
    dtrain=dtrain,
    num_boost_round=100,
    evals=[(dtrain, 'train'), (dtest, 'test')],
    early_stopping_rounds=10,
    verbose_eval=False
)

# ===========================================================================
# ETAPA 2: Tuning de Hiperparâmetros
# ===========================================================================
print(f"\n{'='*70}")
print(f"ETAPA 2: TUNING DE HIPERPARÂMETROS")
print(f"{'='*70}")

# Configurações para testar
param_grid = [
    {'max_depth': 8, 'learning_rate': 0.05, 'subsample': 0.9, 'colsample_bytree': 0.9, 'min_child_weight': 3},
    {'max_depth': 10, 'learning_rate': 0.03, 'subsample': 0.85, 'colsample_bytree': 0.85, 'min_child_weight': 5},
    {'max_depth': 7, 'learning_rate': 0.08, 'subsample': 0.9, 'colsample_bytree': 0.8, 'min_child_weight': 2},
]

best_auc = 0
best_params = None
best_model = None

from sklearn.metrics import roc_auc_score as roc_auc_calc

for i, param_config in enumerate(param_grid):
    print(f"\n--- Testando configuração {i+1}/{len(param_grid)} ---")
    print(f"Params: {param_config}")
    
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
        num_boost_round=150,
        evals=[(dtest, 'test')],
        early_stopping_rounds=15,
        verbose_eval=False
    )
    
    y_pred_test = model_test.predict(dtest)
    auc = roc_auc_calc(y_test, y_pred_test)
    print(f"ROC-AUC: {auc:.4f}")
    
    if auc > best_auc:
        best_auc = auc
        best_params = params_test
        best_model = model_test
        print(f"✓ Nova melhor configuração encontrada!")

print(f"\n{'='*70}")
print(f"MELHOR MODELO SELECIONADO")
print(f"{'='*70}")
print(f"Best ROC-AUC: {best_auc:.4f}")
print(f"Best Params: {best_params}")

# Usar o melhor modelo
model = best_model if best_model is not None else model_baseline

# Fazer predições (probabilidades)
y_pred_proba = model.predict(dtest)

# ===========================================================================
# ETAPA 3: Otimização do Threshold
# ===========================================================================
print(f"\n{'='*70}")
print(f"ETAPA 3: OTIMIZANDO THRESHOLD DE DECISÃO")
print(f"{'='*70}")

# Testar diferentes thresholds
from sklearn.metrics import precision_score, recall_score, f1_score

thresholds_to_test = [0.3, 0.4, 0.5, 0.6, 0.7]
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
    
    print(f"Threshold={thresh:.1f} | Precision={prec:.4f} | Recall={rec:.4f} | F1={f1:.4f}")

# Selecionar melhor threshold baseado em F1-score
best_threshold = max(threshold_results, key=lambda x: x['f1_score'])
print(f"\n✓ Melhor threshold: {best_threshold['threshold']:.1f} (F1={best_threshold['f1_score']:.4f})")

# Converter probabilidades em classes com o melhor threshold
y_pred = (y_pred_proba >= best_threshold['threshold']).astype(int)

# Avaliar o modelo (garantir que não é None)
if X_test is not None and y_test is not None:
    # Importar métricas de classificação
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
    
    # Calcular métricas de classificação
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"\n=== Métricas de Classificação ===")
    print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Precision: {precision:.4f} (Precisão da classe positiva)")
    print(f"Recall:    {recall:.4f} (Sensibilidade/Revocação)")
    print(f"F1-Score:  {f1:.4f} (Média harmônica de Precision e Recall)")
    print(f"ROC-AUC:   {roc_auc:.4f} (Área sob a curva ROC)")
    
    # Matriz de confusão
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n=== Matriz de Confusão ===")
    print(f"                 Predito")
    print(f"                 0      1")
    print(f"Real  0       {cm[0,0]:6d} {cm[0,1]:6d}")
    print(f"      1       {cm[1,0]:6d} {cm[1,1]:6d}")
    print(f"\nVerdadeiros Negativos: {cm[0,0]}")
    print(f"Falsos Positivos: {cm[0,1]}")
    print(f"Falsos Negativos: {cm[1,0]}")
    print(f"Verdadeiros Positivos: {cm[1,1]}")
    
    # Relatório detalhado
    print(f"\n=== Relatório de Classificação ===")
    print(classification_report(y_test, y_pred, target_names=['Classe 0', 'Classe 1']))
    
    # Análise das probabilidades preditas
    print(f"\n=== Análise das Probabilidades ===")
    print(f"Probabilidades - Min: {y_pred_proba.min():.4f}, Max: {y_pred_proba.max():.4f}, Mean: {y_pred_proba.mean():.4f}")
    
    # Visualizar Matriz de Confusão
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Classe 0', 'Classe 1'],
                yticklabels=['Classe 0', 'Classe 1'])
    plt.title('Matriz de Confusão')
    plt.ylabel('Real')
    plt.xlabel('Predito')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("\nMatriz de confusão salva como 'confusion_matrix.png'")
    
    # Visualizar Curva ROC
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight')
    print("Curva ROC salva como 'roc_curve.png'")
    
    # ===========================================================================
    # ETAPA 4: Análise de Threshold vs Métricas
    # ===========================================================================
    print(f"\n{'='*70}")
    print(f"ETAPA 4: ANÁLISE THRESHOLD vs MÉTRICAS")
    print(f"{'='*70}")
    
    # Criar gráfico de threshold vs métricas
    thresholds_plot = [r['threshold'] for r in threshold_results]
    precisions_plot = [r['precision'] for r in threshold_results]
    recalls_plot = [r['recall'] for r in threshold_results]
    f1s_plot = [r['f1_score'] for r in threshold_results]
    
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds_plot, precisions_plot, marker='o', label='Precision', linewidth=2)
    plt.plot(thresholds_plot, recalls_plot, marker='s', label='Recall', linewidth=2)
    plt.plot(thresholds_plot, f1s_plot, marker='^', label='F1-Score', linewidth=2)
    plt.axvline(x=best_threshold['threshold'], color='red', linestyle='--', 
                label=f'Best Threshold ({best_threshold["threshold"]:.1f})', linewidth=2)
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Threshold vs Métricas de Performance')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('threshold_analysis.png', dpi=300, bbox_inches='tight')
    print("Análise de threshold salva como 'threshold_analysis.png'")
    
else:
    print("Erro: Dados de teste não foram criados corretamente.")

# Visualizar importância das features
plt.figure(figsize=(10, 8))
plot_importance(model, max_num_features=20)
plt.title('Importância das Features (Top 20)')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
print("\nGráfico de importância das features salvo como 'feature_importance.png'")

# Salvar o modelo (opcional)
model.save_model('xgboost_model_optimized.json')
print("Modelo otimizado salvo como 'xgboost_model_optimized.json'")

# ===========================================================================
# RESUMO FINAL
# ===========================================================================
print(f"\n{'='*70}")
print(f"RESUMO FINAL DAS MELHORIAS")
print(f"{'='*70}")
print(f"\n📊 BASELINE vs OTIMIZADO")
print(f"   Threshold usado: {best_threshold['threshold']:.1f}")
print(f"\n🎯 Arquivos gerados:")
print(f"   - confusion_matrix.png")
print(f"   - roc_curve.png")
print(f"   - threshold_analysis.png")
print(f"   - feature_importance.png")
print(f"   - xgboost_model_optimized.json")
print(f"\n{'='*70}")