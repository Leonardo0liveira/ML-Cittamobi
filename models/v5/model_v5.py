from google.cloud import bigquery
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_auc_score, accuracy_score, precision_score, 
                             recall_score, f1_score, confusion_matrix, classification_report)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier, 
                              StackingClassifier, VotingClassifier)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb
from collections import Counter
import warnings
import time
warnings.filterwarnings('ignore')

project_id = "proj-ml-469320"
client = bigquery.Client(project=project_id)

# ===========================================================================
# ETAPA 1: CARREGAR DADOS
# ===========================================================================
print(f"\n{'='*70}")
print(f"MODELO V5 - COMPARAÇÃO DE MÚLTIPLOS ALGORITMOS")
print(f"{'='*70}")
print(f"ETAPA 1: CARREGANDO DATASET COM AMOSTRAGEM ALEATÓRIA")
print(f"{'='*70}")

query = """
    SELECT * FROM `proj-ml-469320.app_cittamobi.dataset-updated` 
    TABLESAMPLE SYSTEM (20 PERCENT)
    LIMIT 200000
"""

print("Carregando 200,000 amostras com amostragem aleatória...")
df = client.query(query).to_dataframe()
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
    df['conversion_interaction'] = df['user_conversion_rate'] * df['stop_conversion_rate'] * 10000
    
if 'user_avg_dist' in df.columns and 'dist_device_stop' in df.columns:
    df['dist_deviation'] = np.abs(df['dist_device_stop'] - df['user_avg_dist'])
    df['dist_ratio'] = df['dist_device_stop'] / (df['user_avg_dist'] + 1)

if 'user_frequency' in df.columns and 'stop_event_count_agg' in df.columns:
    df['user_stop_affinity'] = df['user_frequency'] * df['stop_event_count_agg']

print(f"✓ Features de interação criadas")

# ===========================================================================
# ETAPA 3: LIMPEZA MODERADA (IGUAL V4)
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
    q10 = df['user_frequency'].quantile(0.10)
    before = len(df)
    df = df[df['user_frequency'] >= q10]
    print(f"✓ Filtro 1: Removidos {before - len(df):,} registros ({(before - len(df))/df_original_size*100:.1f}%)")

if 'device_lat' in df.columns and 'device_lon' in df.columns:
    before = len(df)
    df = df[(df['device_lat'].between(-90, 90)) & (df['device_lon'].between(-180, 180))]
    print(f"✓ Filtro 2: Removidos {before - len(df):,} registros ({(before - len(df))/df_original_size*100:.1f}%)")

if 'dist_device_stop' in df.columns:
    q98 = df['dist_device_stop'].quantile(0.98)
    before = len(df)
    df = df[df['dist_device_stop'] <= q98]
    print(f"✓ Filtro 3: Removidos {before - len(df):,} registros ({(before - len(df))/df_original_size*100:.1f}%)")

total_removed = 200000 - len(df)
removal_pct = total_removed / 200000 * 100
print(f"\n✓ Total mantido: {len(df):,} ({100-removal_pct:.1f}%)")

# ===========================================================================
# ETAPA 4: PREPARAÇÃO DE FEATURES
# ===========================================================================
print(f"\n{'='*70}")
print(f"ETAPA 4: PREPARAÇÃO DE FEATURES")
print(f"{'='*70}")

features_to_drop = ['y_pred', 'y_pred_proba', 'ctm_service_route', 'direction', 'lotacao_proxy_binaria']
X = df.drop(columns=[target] + features_to_drop, errors='ignore')
y = df[target]

# Processar timestamp
if 'event_timestamp' in X.columns:
    X['event_timestamp'] = pd.to_datetime(X['event_timestamp'])
    X['day'] = X['event_timestamp'].dt.day
    X['hour'] = X['event_timestamp'].dt.hour
    X['dayofweek'] = X['event_timestamp'].dt.dayofweek
    X['minute'] = X['event_timestamp'].dt.minute
    X['week_of_year'] = X['event_timestamp'].dt.isocalendar().week.astype(int)
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

# Label Encoding
categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
if categorical_cols:
    le = LabelEncoder()
    for col in categorical_cols:
        X[col] = le.fit_transform(X[col].astype(str))

X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(0)

print(f"✓ Features totais: {X.shape[1]}")

# ===========================================================================
# ETAPA 5: SELEÇÃO DE FEATURES (TOP 50 - IGUAL V4)
# ===========================================================================
print(f"\n{'='*70}")
print(f"ETAPA 5: SELEÇÃO DAS MELHORES FEATURES")
print(f"{'='*70}")

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

importance_dict = model_temp.get_score(importance_type='gain')
importance_df = pd.DataFrame([
    {'feature': k, 'importance': v} 
    for k, v in importance_dict.items()
]).sort_values('importance', ascending=False)

top_n = 50
top_features = importance_df.head(top_n)['feature'].tolist()
X_selected = X[top_features].copy()

print(f"✓ Features selecionadas: {len(top_features)}")
print(f"✓ Top 10 features:")
for i, row in importance_df.head(10).iterrows():
    print(f"   {i+1}. {row['feature']}: {row['importance']:.2f}")

# ===========================================================================
# ETAPA 6: DIVISÃO TEMPORAL
# ===========================================================================
print(f"\n{'='*70}")
print(f"ETAPA 6: DIVISÃO TEMPORAL (TimeSeriesSplit)")
print(f"{'='*70}")

tscv = TimeSeriesSplit(n_splits=3)
X_train, X_test, y_train, y_test = None, None, None, None

for fold, (train_index, test_index) in enumerate(tscv.split(X_selected)):
    X_train, X_test = X_selected.iloc[train_index], X_selected.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    if fold == 2:
        print(f"Fold {fold + 1}: Train={len(X_train):,} | Test={len(X_test):,}")

print(f"\n=== Distribuição Original do Treino ===")
train_dist = y_train.value_counts()
print(f"Classe 0: {train_dist[0]:,} ({train_dist[0]/len(y_train)*100:.2f}%)")
print(f"Classe 1: {train_dist[1]:,} ({train_dist[1]/len(y_train)*100:.2f}%)")
print(f"Razão (0/1): {train_dist[0]/train_dist[1]:.2f}:1")

# ===========================================================================
# ETAPA 7: NORMALIZAÇÃO (IMPORTANTE PARA ALGUNS MODELOS)
# ===========================================================================
print(f"\n{'='*70}")
print(f"ETAPA 7: NORMALIZAÇÃO DOS DADOS")
print(f"{'='*70}")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Converter de volta para DataFrame para manter nomes
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

print(f"✓ Dados normalizados (StandardScaler)")

# ===========================================================================
# ETAPA 8: TESTANDO MÚLTIPLOS ALGORITMOS
# ===========================================================================
print(f"\n{'='*70}")
print(f"ETAPA 8: TESTANDO MÚLTIPLOS ALGORITMOS")
print(f"{'='*70}")

results = []

# ===========================================================================
# MODELO 1: Logistic Regression (L2) com class_weight + Calibração
# ===========================================================================
print(f"\n--- Modelo 1: LOGISTIC REGRESSION (L2 + BALANCED + CALIBRAÇÃO) ---")
start_time = time.time()

lr_base = LogisticRegression(
    penalty='l2',
    C=1.0,
    class_weight='balanced',
    max_iter=1000,
    solver='lbfgs',
    random_state=42,
    n_jobs=-1
)

# Calibração de probabilidades
lr_model = CalibratedClassifierCV(lr_base, method='sigmoid', cv=3)
lr_model.fit(X_train_scaled, y_train)

y_pred_proba_lr = lr_model.predict_proba(X_test_scaled)[:, 1]
y_pred_lr = (y_pred_proba_lr >= 0.5).astype(int)

train_time = time.time() - start_time

results.append({
    'model': 'Logistic Regression (L2 + Calibrated)',
    'roc_auc': roc_auc_score(y_test, y_pred_proba_lr),
    'accuracy': accuracy_score(y_test, y_pred_lr),
    'precision': precision_score(y_test, y_pred_lr, zero_division=0),
    'recall': recall_score(y_test, y_pred_lr),
    'f1_score': f1_score(y_test, y_pred_lr),
    'f1_macro': f1_score(y_test, y_pred_lr, average='macro'),
    'train_time': train_time,
    'y_pred_proba': y_pred_proba_lr,
    'classifier': lr_model
})

print(f"ROC-AUC: {results[-1]['roc_auc']:.4f} | F1-Macro: {results[-1]['f1_macro']:.4f} | Tempo: {train_time:.2f}s")

# ===========================================================================
# MODELO 2: SGDClassifier
# ===========================================================================
print(f"\n--- Modelo 2: SGD CLASSIFIER ---")
start_time = time.time()

sgd_model = SGDClassifier(
    loss='log_loss',  # Logistic regression
    penalty='l2',
    alpha=0.0001,
    class_weight='balanced',
    max_iter=1000,
    tol=1e-3,
    random_state=42,
    n_jobs=-1
)

sgd_model.fit(X_train_scaled, y_train)

# SGD não tem predict_proba por padrão, usar decision_function
y_pred_decision = sgd_model.decision_function(X_test_scaled)
# Normalizar para [0, 1]
y_pred_proba_sgd = 1 / (1 + np.exp(-y_pred_decision))
y_pred_sgd = sgd_model.predict(X_test_scaled)

train_time = time.time() - start_time

results.append({
    'model': 'SGD Classifier',
    'roc_auc': roc_auc_score(y_test, y_pred_proba_sgd),
    'accuracy': accuracy_score(y_test, y_pred_sgd),
    'precision': precision_score(y_test, y_pred_sgd, zero_division=0),
    'recall': recall_score(y_test, y_pred_sgd),
    'f1_score': f1_score(y_test, y_pred_sgd),
    'f1_macro': f1_score(y_test, y_pred_sgd, average='macro'),
    'train_time': train_time,
    'y_pred_proba': y_pred_proba_sgd,
    'classifier': sgd_model
})

print(f"ROC-AUC: {results[-1]['roc_auc']:.4f} | F1-Macro: {results[-1]['f1_macro']:.4f} | Tempo: {train_time:.2f}s")

# ===========================================================================
# MODELO 3: Linear SVM (LinearSVC)
# ===========================================================================
print(f"\n--- Modelo 3: LINEAR SVM ---")
start_time = time.time()

svm_model = LinearSVC(
    penalty='l2',
    C=1.0,
    class_weight='balanced',
    max_iter=1000,
    random_state=42
)

svm_model.fit(X_train_scaled, y_train)

y_pred_decision_svm = svm_model.decision_function(X_test_scaled)
y_pred_proba_svm = 1 / (1 + np.exp(-y_pred_decision_svm))
y_pred_svm = svm_model.predict(X_test_scaled)

train_time = time.time() - start_time

results.append({
    'model': 'Linear SVM',
    'roc_auc': roc_auc_score(y_test, y_pred_proba_svm),
    'accuracy': accuracy_score(y_test, y_pred_svm),
    'precision': precision_score(y_test, y_pred_svm, zero_division=0),
    'recall': recall_score(y_test, y_pred_svm),
    'f1_score': f1_score(y_test, y_pred_svm),
    'f1_macro': f1_score(y_test, y_pred_svm, average='macro'),
    'train_time': train_time,
    'y_pred_proba': y_pred_proba_svm,
    'classifier': svm_model
})

print(f"ROC-AUC: {results[-1]['roc_auc']:.4f} | F1-Macro: {results[-1]['f1_macro']:.4f} | Tempo: {train_time:.2f}s")

# ===========================================================================
# MODELO 4: Radial SVM (RBF kernel)
# ===========================================================================
print(f"\n--- Modelo 4: RADIAL SVM (RBF KERNEL) ---")
start_time = time.time()

# Usar amostra menor para RBF SVM (muito lento em datasets grandes)
sample_size = min(30000, len(X_train_scaled))
sample_indices = np.random.choice(len(X_train_scaled), size=sample_size, replace=False)
X_train_sample = X_train_scaled.iloc[sample_indices]
y_train_sample = y_train.iloc[sample_indices]

print(f"Usando amostra de {sample_size:,} registros para treino (RBF é computacionalmente intensivo)")

rbf_svm_model = SVC(
    kernel='rbf',
    C=1.0,
    gamma='scale',
    class_weight='balanced',
    probability=True,  # Necessário para predict_proba
    random_state=42,
    max_iter=1000
)

rbf_svm_model.fit(X_train_sample, y_train_sample)

y_pred_proba_rbf = rbf_svm_model.predict_proba(X_test_scaled)[:, 1]
y_pred_rbf = rbf_svm_model.predict(X_test_scaled)

train_time = time.time() - start_time

results.append({
    'model': 'Radial SVM (RBF)',
    'roc_auc': roc_auc_score(y_test, y_pred_proba_rbf),
    'accuracy': accuracy_score(y_test, y_pred_rbf),
    'precision': precision_score(y_test, y_pred_rbf, zero_division=0),
    'recall': recall_score(y_test, y_pred_rbf),
    'f1_score': f1_score(y_test, y_pred_rbf),
    'f1_macro': f1_score(y_test, y_pred_rbf, average='macro'),
    'train_time': train_time,
    'y_pred_proba': y_pred_proba_rbf,
    'classifier': rbf_svm_model
})

print(f"ROC-AUC: {results[-1]['roc_auc']:.4f} | F1-Macro: {results[-1]['f1_macro']:.4f} | Tempo: {train_time:.2f}s")

# ===========================================================================
# MODELO 5: Decision Tree
# ===========================================================================
print(f"\n--- Modelo 5: DECISION TREE ---")
start_time = time.time()

dt_model = DecisionTreeClassifier(
    max_depth=15,
    min_samples_split=100,
    min_samples_leaf=50,
    class_weight='balanced',
    random_state=42
)

dt_model.fit(X_train, y_train)

y_pred_proba_dt = dt_model.predict_proba(X_test)[:, 1]
y_pred_dt = dt_model.predict(X_test)

train_time = time.time() - start_time

results.append({
    'model': 'Decision Tree',
    'roc_auc': roc_auc_score(y_test, y_pred_proba_dt),
    'accuracy': accuracy_score(y_test, y_pred_dt),
    'precision': precision_score(y_test, y_pred_dt, zero_division=0),
    'recall': recall_score(y_test, y_pred_dt),
    'f1_score': f1_score(y_test, y_pred_dt),
    'f1_macro': f1_score(y_test, y_pred_dt, average='macro'),
    'train_time': train_time,
    'y_pred_proba': y_pred_proba_dt,
    'classifier': dt_model
})

print(f"ROC-AUC: {results[-1]['roc_auc']:.4f} | F1-Macro: {results[-1]['f1_macro']:.4f} | Tempo: {train_time:.2f}s")

# ===========================================================================
# MODELO 6: Random Forest
# ===========================================================================
print(f"\n--- Modelo 6: RANDOM FOREST ---")
start_time = time.time()

rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    min_samples_split=100,
    min_samples_leaf=50,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

y_pred_proba_rf = rf_model.predict_proba(X_test)[:, 1]
y_pred_rf = rf_model.predict(X_test)

train_time = time.time() - start_time

results.append({
    'model': 'Random Forest',
    'roc_auc': roc_auc_score(y_test, y_pred_proba_rf),
    'accuracy': accuracy_score(y_test, y_pred_rf),
    'precision': precision_score(y_test, y_pred_rf, zero_division=0),
    'recall': recall_score(y_test, y_pred_rf),
    'f1_score': f1_score(y_test, y_pred_rf),
    'f1_macro': f1_score(y_test, y_pred_rf, average='macro'),
    'train_time': train_time,
    'y_pred_proba': y_pred_proba_rf,
    'classifier': rf_model
})

print(f"ROC-AUC: {results[-1]['roc_auc']:.4f} | F1-Macro: {results[-1]['f1_macro']:.4f} | Tempo: {train_time:.2f}s")

# ===========================================================================
# MODELO 7: Extra Trees
# ===========================================================================
print(f"\n--- Modelo 7: EXTRA TREES ---")
start_time = time.time()

et_model = ExtraTreesClassifier(
    n_estimators=100,
    max_depth=15,
    min_samples_split=100,
    min_samples_leaf=50,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

et_model.fit(X_train, y_train)

y_pred_proba_et = et_model.predict_proba(X_test)[:, 1]
y_pred_et = et_model.predict(X_test)

train_time = time.time() - start_time

results.append({
    'model': 'Extra Trees',
    'roc_auc': roc_auc_score(y_test, y_pred_proba_et),
    'accuracy': accuracy_score(y_test, y_pred_et),
    'precision': precision_score(y_test, y_pred_et, zero_division=0),
    'recall': recall_score(y_test, y_pred_et),
    'f1_score': f1_score(y_test, y_pred_et),
    'f1_macro': f1_score(y_test, y_pred_et, average='macro'),
    'train_time': train_time,
    'y_pred_proba': y_pred_proba_et,
    'classifier': et_model
})

print(f"ROC-AUC: {results[-1]['roc_auc']:.4f} | F1-Macro: {results[-1]['f1_macro']:.4f} | Tempo: {train_time:.2f}s")

# ===========================================================================
# MODELO 8: k-NN com weights='distance'
# ===========================================================================
print(f"\n--- Modelo 8: K-NEAREST NEIGHBORS ---")
start_time = time.time()

knn_model = KNeighborsClassifier(
    n_neighbors=50,
    weights='distance',
    algorithm='auto',
    n_jobs=-1
)

knn_model.fit(X_train_scaled, y_train)

y_pred_proba_knn = knn_model.predict_proba(X_test_scaled)[:, 1]
y_pred_knn = knn_model.predict(X_test_scaled)

train_time = time.time() - start_time

results.append({
    'model': 'k-NN (distance weighted)',
    'roc_auc': roc_auc_score(y_test, y_pred_proba_knn),
    'accuracy': accuracy_score(y_test, y_pred_knn),
    'precision': precision_score(y_test, y_pred_knn, zero_division=0),
    'recall': recall_score(y_test, y_pred_knn),
    'f1_score': f1_score(y_test, y_pred_knn),
    'f1_macro': f1_score(y_test, y_pred_knn, average='macro'),
    'train_time': train_time,
    'y_pred_proba': y_pred_proba_knn,
    'classifier': knn_model
})

print(f"ROC-AUC: {results[-1]['roc_auc']:.4f} | F1-Macro: {results[-1]['f1_macro']:.4f} | Tempo: {train_time:.2f}s")

# ===========================================================================
# MODELO 9: Gaussian Naive Bayes
# ===========================================================================
print(f"\n--- Modelo 9: GAUSSIAN NAIVE BAYES ---")
start_time = time.time()

gnb_model = GaussianNB()
gnb_model.fit(X_train_scaled, y_train)

y_pred_proba_gnb = gnb_model.predict_proba(X_test_scaled)[:, 1]
y_pred_gnb = gnb_model.predict(X_test_scaled)

train_time = time.time() - start_time

results.append({
    'model': 'Gaussian Naive Bayes',
    'roc_auc': roc_auc_score(y_test, y_pred_proba_gnb),
    'accuracy': accuracy_score(y_test, y_pred_gnb),
    'precision': precision_score(y_test, y_pred_gnb, zero_division=0),
    'recall': recall_score(y_test, y_pred_gnb),
    'f1_score': f1_score(y_test, y_pred_gnb),
    'f1_macro': f1_score(y_test, y_pred_gnb, average='macro'),
    'train_time': train_time,
    'y_pred_proba': y_pred_proba_gnb,
    'classifier': gnb_model
})

print(f"ROC-AUC: {results[-1]['roc_auc']:.4f} | F1-Macro: {results[-1]['f1_macro']:.4f} | Tempo: {train_time:.2f}s")

# ===========================================================================
# MODELO 10: Bernoulli Naive Bayes
# ===========================================================================
print(f"\n--- Modelo 10: BERNOULLI NAIVE BAYES ---")
start_time = time.time()

# Converter dados para binário (0 ou 1) para BernoulliNB
X_train_binary = (X_train_scaled > 0).astype(int)
X_test_binary = (X_test_scaled > 0).astype(int)

bnb_model = BernoulliNB()
bnb_model.fit(X_train_binary, y_train)

y_pred_proba_bnb = bnb_model.predict_proba(X_test_binary)[:, 1]
y_pred_bnb = bnb_model.predict(X_test_binary)

train_time = time.time() - start_time

results.append({
    'model': 'Bernoulli Naive Bayes',
    'roc_auc': roc_auc_score(y_test, y_pred_proba_bnb),
    'accuracy': accuracy_score(y_test, y_pred_bnb),
    'precision': precision_score(y_test, y_pred_bnb, zero_division=0),
    'recall': recall_score(y_test, y_pred_bnb),
    'f1_score': f1_score(y_test, y_pred_bnb),
    'f1_macro': f1_score(y_test, y_pred_bnb, average='macro'),
    'train_time': train_time,
    'y_pred_proba': y_pred_proba_bnb,
    'classifier': bnb_model
})

print(f"ROC-AUC: {results[-1]['roc_auc']:.4f} | F1-Macro: {results[-1]['f1_macro']:.4f} | Tempo: {train_time:.2f}s")

# ===========================================================================
# MODELO 11: XGBoost (Referência V4)
# ===========================================================================
print(f"\n--- Modelo 11: XGBOOST (REFERÊNCIA V4) ---")
start_time = time.time()

scale_pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()

xgb_params = {
    'objective': 'binary:logistic',
    'max_depth': 18,
    'learning_rate': 0.02,
    'min_child_weight': 3,
    'subsample': 0.9,
    'colsample_bytree': 0.9,
    'scale_pos_weight': scale_pos_weight,
    'eval_metric': 'logloss',
    'seed': 42
}

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

xgb_model = xgb.train(
    params=xgb_params,
    dtrain=dtrain,
    num_boost_round=250,
    evals=[(dtest, 'test')],
    early_stopping_rounds=25,
    verbose_eval=False
)

y_pred_proba_xgb = xgb_model.predict(dtest)
y_pred_xgb = (y_pred_proba_xgb >= 0.5).astype(int)

train_time = time.time() - start_time

results.append({
    'model': 'XGBoost (V4 Reference)',
    'roc_auc': roc_auc_score(y_test, y_pred_proba_xgb),
    'accuracy': accuracy_score(y_test, y_pred_xgb),
    'precision': precision_score(y_test, y_pred_xgb, zero_division=0),
    'recall': recall_score(y_test, y_pred_xgb),
    'f1_score': f1_score(y_test, y_pred_xgb),
    'f1_macro': f1_score(y_test, y_pred_xgb, average='macro'),
    'train_time': train_time,
    'y_pred_proba': y_pred_proba_xgb,
    'classifier': xgb_model
})

print(f"ROC-AUC: {results[-1]['roc_auc']:.4f} | F1-Macro: {results[-1]['f1_macro']:.4f} | Tempo: {train_time:.2f}s")

# ===========================================================================
# MODELO 12: Stacking Classifier
# ===========================================================================
print(f"\n--- Modelo 12: STACKING CLASSIFIER ---")
print("Treinando ensemble com os 3 melhores modelos base...")
start_time = time.time()

# Selecionar os 3 melhores modelos (excluindo XGBoost e Stacking)
sorted_results = sorted([r for r in results if 'XGBoost' not in r['model']], 
                       key=lambda x: x['f1_macro'], reverse=True)[:3]

# Criar NOVOS estimadores (não reutilizar os já treinados)
base_estimators = []
for r in sorted_results:
    model_name = r['model']
    if 'k-NN' in model_name:
        base_estimators.append(('kNN', KNeighborsClassifier(n_neighbors=50, weights='distance', n_jobs=-1)))
    elif 'Extra Trees' in model_name:
        base_estimators.append(('ExtraTrees', ExtraTreesClassifier(n_estimators=100, max_depth=15, 
                                                                     min_samples_split=100, min_samples_leaf=50,
                                                                     class_weight='balanced', random_state=42, n_jobs=-1)))
    elif 'Gaussian' in model_name:
        base_estimators.append(('GaussianNB', GaussianNB()))
    elif 'Decision Tree' in model_name:
        base_estimators.append(('DecisionTree', DecisionTreeClassifier(max_depth=15, min_samples_split=100,
                                                                        min_samples_leaf=50, class_weight='balanced',
                                                                        random_state=42)))
    elif 'Random Forest' in model_name:
        base_estimators.append(('RandomForest', RandomForestClassifier(n_estimators=100, max_depth=15,
                                                                        min_samples_split=100, min_samples_leaf=50,
                                                                        class_weight='balanced', random_state=42, n_jobs=-1)))

print(f"Base estimators: {[name for name, _ in base_estimators]}")

stacking_model = StackingClassifier(
    estimators=base_estimators,
    final_estimator=LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000),
    cv=3,
    n_jobs=1  # Evitar problemas de paralelização
)

# Converter para array numpy com cópia para evitar problemas de writeable
X_train_scaled_copy = np.array(X_train_scaled, copy=True)
y_train_copy = np.array(y_train, copy=True)

stacking_model.fit(X_train_scaled_copy, y_train_copy)

X_test_scaled_copy = np.array(X_test_scaled, copy=True)
y_pred_proba_stack = stacking_model.predict_proba(X_test_scaled_copy)[:, 1]
y_pred_stack = stacking_model.predict(X_test_scaled_copy)

train_time = time.time() - start_time

results.append({
    'model': 'Stacking (Top 3 + LR)',
    'roc_auc': roc_auc_score(y_test, y_pred_proba_stack),
    'accuracy': accuracy_score(y_test, y_pred_stack),
    'precision': precision_score(y_test, y_pred_stack, zero_division=0),
    'recall': recall_score(y_test, y_pred_stack),
    'f1_score': f1_score(y_test, y_pred_stack),
    'f1_macro': f1_score(y_test, y_pred_stack, average='macro'),
    'train_time': train_time,
    'y_pred_proba': y_pred_proba_stack,
    'classifier': stacking_model
})

print(f"ROC-AUC: {results[-1]['roc_auc']:.4f} | F1-Macro: {results[-1]['f1_macro']:.4f} | Tempo: {train_time:.2f}s")

# ===========================================================================
# ETAPA 9: COMPARAÇÃO E ANÁLISE
# ===========================================================================
print(f"\n{'='*70}")
print(f"ETAPA 9: COMPARAÇÃO DE TODOS OS MODELOS")
print(f"{'='*70}")

results_df = pd.DataFrame([{
    'Modelo': r['model'],
    'ROC-AUC': f"{r['roc_auc']:.4f}",
    'Accuracy': f"{r['accuracy']:.4f}",
    'Precision': f"{r['precision']:.4f}",
    'Recall': f"{r['recall']:.4f}",
    'F1-Score': f"{r['f1_score']:.4f}",
    'F1-Macro': f"{r['f1_macro']:.4f}",
    'Tempo (s)': f"{r['train_time']:.2f}"
} for r in results])

print("\n" + results_df.to_string(index=False))

# Selecionar melhor modelo
best_idx = np.argmax([r['f1_macro'] for r in results])
best_result = results[best_idx]

print(f"\n{'='*70}")
print(f"🏆 MELHOR MODELO: {best_result['model']}")
print(f"{'='*70}")
print(f"ROC-AUC:   {best_result['roc_auc']:.4f}")
print(f"Accuracy:  {best_result['accuracy']:.4f}")
print(f"Precision: {best_result['precision']:.4f}")
print(f"Recall:    {best_result['recall']:.4f}")
print(f"F1-Score:  {best_result['f1_score']:.4f}")
print(f"F1-Macro:  {best_result['f1_macro']:.4f} ⭐")
print(f"Tempo:     {best_result['train_time']:.2f}s")

# ===========================================================================
# ETAPA 10: VISUALIZAÇÕES
# ===========================================================================
print(f"\n{'='*70}")
print(f"ETAPA 10: GERANDO VISUALIZAÇÕES")
print(f"{'='*70}")

# Comparação de métricas
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

metrics = ['roc_auc', 'precision', 'recall', 'f1_macro']
titles = ['ROC-AUC', 'Precision', 'Recall', 'F1-Macro']
colors = plt.cm.viridis(np.linspace(0, 1, len(results)))

for idx, (metric, title) in enumerate(zip(metrics, titles)):
    ax = axes[idx // 2, idx % 2]
    
    values = [r[metric] for r in results]
    models = [r['model'] for r in results]
    
    bars = ax.barh(range(len(models)), values, color=colors)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=9)
    ax.set_xlabel(title, fontsize=11, fontweight='bold')
    ax.set_title(f'{title} por Modelo', fontsize=12, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # Destacar o melhor
    best_val_idx = np.argmax(values)
    bars[best_val_idx].set_color('#FF6B6B')
    
    # Adicionar valores
    for i, v in enumerate(values):
        ax.text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=8)

plt.tight_layout()
plt.savefig('v5_models_comparison.png', dpi=300, bbox_inches='tight')
print("✓ v5_models_comparison.png")

# Tempo de treinamento
fig, ax = plt.subplots(figsize=(12, 6))
times = [r['train_time'] for r in results]
models = [r['model'] for r in results]

bars = ax.bar(range(len(models)), times, color=colors)
ax.set_xticks(range(len(models)))
ax.set_xticklabels(models, rotation=45, ha='right', fontsize=9)
ax.set_ylabel('Tempo de Treinamento (segundos)', fontsize=11, fontweight='bold')
ax.set_title('Tempo de Treinamento por Modelo', fontsize=12, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

for i, v in enumerate(times):
    ax.text(i, v + 0.5, f'{v:.1f}s', ha='center', fontsize=8)

plt.tight_layout()
plt.savefig('v5_training_time.png', dpi=300, bbox_inches='tight')
print("✓ v5_training_time.png")

# Matriz de confusão do melhor modelo
best_y_pred = (best_result['y_pred_proba'] >= 0.5).astype(int)
cm = confusion_matrix(y_test, best_y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Classe 0', 'Classe 1'],
            yticklabels=['Classe 0', 'Classe 1'])
plt.title(f'Matriz de Confusão - {best_result["model"]}')
plt.ylabel('Real')
plt.xlabel('Predito')
plt.tight_layout()
plt.savefig('v5_confusion_matrix_best.png', dpi=300, bbox_inches='tight')
print("✓ v5_confusion_matrix_best.png")

# ===========================================================================
# RESUMO FINAL
# ===========================================================================
print(f"\n{'='*70}")
print(f"🎉 V5 - RESUMO FINAL")
print(f"{'='*70}")

print(f"\n📊 Modelos Testados: {len(results)}")
print(f"1. Logistic Regression (L2 + Calibrated)")
print(f"2. SGD Classifier")
print(f"3. Linear SVM")
print(f"4. Radial SVM (RBF)")
print(f"5. Decision Tree")
print(f"6. Random Forest")
print(f"7. Extra Trees")
print(f"8. k-NN (distance weighted)")
print(f"9. Gaussian Naive Bayes")
print(f"10. Bernoulli Naive Bayes")
print(f"11. XGBoost (V4 Reference)")
print(f"12. Stacking Classifier")

print(f"\n🏆 Melhor Modelo: {best_result['model']}")
print(f"   ROC-AUC: {best_result['roc_auc']:.4f}")
print(f"   F1-Macro: {best_result['f1_macro']:.4f}")

# Comparação com V4
v4_roc_auc = 0.9731
v4_f1_macro = 0.7760

if best_result['roc_auc'] > v4_roc_auc:
    improvement_roc = ((best_result['roc_auc'] - v4_roc_auc) / v4_roc_auc) * 100
    print(f"\n✅ Melhor que V4 em ROC-AUC: +{improvement_roc:.2f}%")
else:
    decline_roc = ((v4_roc_auc - best_result['roc_auc']) / v4_roc_auc) * 100
    print(f"\n⚠️ V4 ainda superior em ROC-AUC: -{decline_roc:.2f}%")

if best_result['f1_macro'] > v4_f1_macro:
    improvement_f1 = ((best_result['f1_macro'] - v4_f1_macro) / v4_f1_macro) * 100
    print(f"✅ Melhor que V4 em F1-Macro: +{improvement_f1:.2f}%")
else:
    decline_f1 = ((v4_f1_macro - best_result['f1_macro']) / v4_f1_macro) * 100
    print(f"⚠️ V4 ainda superior em F1-Macro: -{decline_f1:.2f}%")

print(f"\n📈 Top 3 Modelos por F1-Macro:")
sorted_by_f1 = sorted(results, key=lambda x: x['f1_macro'], reverse=True)
for i, r in enumerate(sorted_by_f1[:3], 1):
    print(f"{i}. {r['model']}: {r['f1_macro']:.4f}")

print(f"\n⏱️ Modelo Mais Rápido: {min(results, key=lambda x: x['train_time'])['model']}")
print(f"   Tempo: {min(results, key=lambda x: x['train_time'])['train_time']:.2f}s")

print(f"\n{'='*70}")

