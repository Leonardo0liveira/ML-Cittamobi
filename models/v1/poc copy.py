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
features_to_drop = ['y_pred', 'y_pred_proba', 'ctm_service_route', 'direction']
print(f"\n=== Removendo features suspeitas ===")
print(f"Features removidas: {features_to_drop}")

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

# Parâmetros do modelo XGBoost
params = {
    'objective': 'reg:squarederror',  # Para regressão (ajuste se for classificação)
    'max_depth': 6,
    'learning_rate': 0.1,
    'n_estimators': 100,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'eval_metric': 'rmse',
    'seed': 42
}

# Treinar o modelo
model = xgb.train(
    params=params,
    dtrain=dtrain,
    num_boost_round=100,
    evals=[(dtrain, 'train'), (dtest, 'test')],
    early_stopping_rounds=10,
    verbose_eval=10
)

# Fazer predições
y_pred = model.predict(dtest)

# Avaliar o modelo (garantir que não é None)
if X_test is not None and y_test is not None:
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f"\n=== Métricas de Avaliação ===")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    
    # Análise adicional de predições
    print(f"\n=== Análise das Predições ===")
    print(f"Predições - Min: {y_pred.min():.4f}, Max: {y_pred.max():.4f}, Mean: {y_pred.mean():.4f}")
    print(f"Real - Min: {y_test.min():.4f}, Max: {y_test.max():.4f}, Mean: {y_test.mean():.4f}")
    
    # Calcular R² score
    from sklearn.metrics import r2_score
    r2 = r2_score(y_test, y_pred)
    print(f"R² Score: {r2:.4f}")
else:
    print("Erro: Dados de teste não foram criados corretamente.")

# Visualizar importância das features
plt.figure(figsize=(10, 8))
plot_importance(model, max_num_features=20)
plt.title('Importância das Features')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
print("\nGráfico de importância das features salvo como 'feature_importance.png'")

# Salvar o modelo (opcional)
model.save_model('xgboost_model.json')
print("Modelo salvo como 'xgboost_model.json'")