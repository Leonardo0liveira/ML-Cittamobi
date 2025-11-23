# 🚀 Guia de Uso do Modelo XGBoost Otimizado

## 📦 Como Carregar e Usar o Modelo

### **1. Carregar o Modelo Treinado**

```python
import xgboost as xgb
import pandas as pd
import numpy as np

# Carregar o modelo otimizado
model = xgb.Booster()
model.load_model('xgboost_model_optimized.json')
```

---

### **2. Preparar Novos Dados**

```python
# Supondo que você tem um DataFrame 'df_novo' com as mesmas colunas do treinamento

# Remover features que foram excluídas no treinamento
features_to_drop = ['y_pred', 'y_pred_proba', 'ctm_service_route', 
                    'direction', 'lotacao_proxy_binaria', 'target']

X_novo = df_novo.drop(columns=features_to_drop, errors='ignore')

# Processar event_timestamp (se existir)
if 'event_timestamp' in X_novo.columns:
    X_novo['year'] = pd.to_datetime(X_novo['event_timestamp']).dt.year
    X_novo['month'] = pd.to_datetime(X_novo['event_timestamp']).dt.month
    X_novo['day'] = pd.to_datetime(X_novo['event_timestamp']).dt.day
    X_novo['hour'] = pd.to_datetime(X_novo['event_timestamp']).dt.hour
    X_novo['dayofweek'] = pd.to_datetime(X_novo['event_timestamp']).dt.dayofweek
    X_novo = X_novo.drop(columns=['event_timestamp'])

# Label Encoding para colunas categóricas
from sklearn.preprocessing import LabelEncoder

categorical_cols = X_novo.select_dtypes(include=['object', 'category']).columns.tolist()
for col in categorical_cols:
    le = LabelEncoder()
    X_novo[col] = le.fit_transform(X_novo[col].astype(str))

# Tratar valores infinitos e NaN
X_novo = X_novo.replace([np.inf, -np.inf], np.nan)
X_novo = X_novo.fillna(0)
```

---

### **3. Fazer Predições**

```python
# Criar DMatrix (formato otimizado do XGBoost)
dmatrix_novo = xgb.DMatrix(X_novo)

# Obter probabilidades
probabilidades = model.predict(dmatrix_novo)

# Converter para classes usando threshold otimizado (0.6)
THRESHOLD_OTIMIZADO = 0.6
predicoes = (probabilidades >= THRESHOLD_OTIMIZADO).astype(int)

# Adicionar resultados ao DataFrame
df_novo['probabilidade_classe_1'] = probabilidades
df_novo['predicao'] = predicoes
```

---

### **4. Interpretar os Resultados**

```python
# Exemplo de interpretação
for i, (prob, pred) in enumerate(zip(probabilidades[:5], predicoes[:5])):
    confianca = prob if pred == 1 else (1 - prob)
    print(f"Amostra {i+1}:")
    print(f"  Predição: Classe {pred}")
    print(f"  Probabilidade: {prob:.2%}")
    print(f"  Confiança: {confianca:.2%}")
    print()
```

**Saída esperada:**
```
Amostra 1:
  Predição: Classe 0
  Probabilidade: 15.32%
  Confiança: 84.68%

Amostra 2:
  Predição: Classe 1
  Probabilidade: 73.45%
  Confiança: 73.45%
```

---

## 🎯 Thresholds Recomendados por Cenário

### **Threshold = 0.6 (Padrão - Equilibrado)**
- **Precision:** 45.19%
- **Recall:** 51.21%
- **F1-Score:** 0.4801
- **Uso:** Equilíbrio entre precision e recall

### **Threshold = 0.7 (Alta Precision)**
- **Precision:** 52.41%
- **Recall:** 40.79%
- **Uso:** Quando falsos positivos são muito custosos

### **Threshold = 0.4 (Alto Recall)**
- **Precision:** 31.28%
- **Recall:** 69.88%
- **Uso:** Quando é crítico não perder casos positivos

### **Como Ajustar:**

```python
# Para usar threshold diferente
THRESHOLD_CUSTOM = 0.7
predicoes_custom = (probabilidades >= THRESHOLD_CUSTOM).astype(int)
```

---

## 📊 Métricas de Performance Esperadas

### **Em Dados de Teste:**
```
Accuracy:  89.02%
Precision: 45.19%
Recall:    51.21%
F1-Score:  0.4801
ROC-AUC:   0.8367
```

### **Interpretação:**
- ✅ 89% das predições estão corretas
- ✅ Quando prediz classe 1, há 45% de chance de estar correto
- ✅ Captura 51% de todos os casos reais da classe 1
- ✅ ROC-AUC de 0.84 indica excelente capacidade discriminativa

---

## 🔍 Features Mais Importantes

O modelo se baseia principalmente nestas features:

1. **stop_event_rate** - Taxa de eventos no ponto
2. **stop_total_samples** - Total de amostras
3. **stop_event_count** - Contagem de eventos
4. **hour** - Hora do dia
5. **hour_sin** - Componente cíclico da hora

**Dica:** Garanta que essas features estejam corretamente calculadas nos novos dados!

---

## ⚙️ Parâmetros do Modelo Otimizado

```python
{
    'objective': 'binary:logistic',
    'max_depth': 10,
    'learning_rate': 0.03,
    'subsample': 0.85,
    'colsample_bytree': 0.85,
    'min_child_weight': 5,
    'scale_pos_weight': 9.31,
    'eval_metric': 'logloss',
    'seed': 42
}
```

---

## 🐛 Troubleshooting

### **Erro: "feature_names mismatch"**
**Causa:** Número ou ordem de features diferente do treinamento.

**Solução:**
```python
# Garantir mesma ordem de colunas
feature_names = model.feature_names
X_novo = X_novo[feature_names]
```

---

### **Erro: "cannot convert float NaN to integer"**
**Causa:** Valores NaN não tratados.

**Solução:**
```python
X_novo = X_novo.fillna(0)
```

---

### **Performance abaixo do esperado**
**Possíveis causas:**
1. **Data drift** - Distribuição dos dados mudou
2. **Features faltando** - Verifique se todas as features estão presentes
3. **Encoding diferente** - Certifique-se de usar o mesmo Label Encoder

**Ação:**
```python
# Verificar features
print("Features esperadas:", model.feature_names)
print("Features disponíveis:", X_novo.columns.tolist())
```

---

## 📈 Monitoramento em Produção

### **1. Métricas a Acompanhar:**

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

# Calcular métricas periodicamente
accuracy = accuracy_score(y_real, predicoes)
precision = precision_score(y_real, predicoes)
recall = recall_score(y_real, predicoes)
roc_auc = roc_auc_score(y_real, probabilidades)

# Alertar se métricas caírem abaixo dos limiares
if roc_auc < 0.80:
    print("⚠️ ALERTA: ROC-AUC abaixo de 0.80 - Considerar retreinamento")
```

---

### **2. Detectar Data Drift:**

```python
# Comparar distribuição de features importantes
import matplotlib.pyplot as plt

# Feature: stop_event_rate
plt.figure(figsize=(10, 5))
plt.hist(X_treino['stop_event_rate'], bins=50, alpha=0.5, label='Treino')
plt.hist(X_novo['stop_event_rate'], bins=50, alpha=0.5, label='Novo')
plt.legend()
plt.title('Distribuição: stop_event_rate')
plt.show()
```

---

### **3. Quando Retreinar:**

✅ **Retreinar se:**
- ROC-AUC cair abaixo de 0.80
- Accuracy cair abaixo de 85%
- Distribuição das features mudou significativamente
- Passou mais de 2 meses desde o último treinamento

---

## 💾 Salvar Predições

```python
# Salvar resultados em CSV
resultado = pd.DataFrame({
    'id': df_novo['id_campo'],  # Ajuste para seu ID
    'probabilidade': probabilidades,
    'predicao': predicoes,
    'confianca': np.where(predicoes == 1, probabilidades, 1 - probabilidades)
})

resultado.to_csv('predicoes_modelo.csv', index=False)
print(f"✅ Predições salvas: {len(resultado)} registros")
```

---

## 🔄 Pipeline Completo de Predição

```python
def predict_pipeline(df_input, model_path='xgboost_model_optimized.json', 
                     threshold=0.6):
    """
    Pipeline completo para fazer predições com o modelo otimizado.
    
    Args:
        df_input: DataFrame com os dados de entrada
        model_path: Caminho para o modelo salvo
        threshold: Threshold de decisão (padrão: 0.6)
    
    Returns:
        DataFrame com predições e probabilidades
    """
    import xgboost as xgb
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import LabelEncoder
    
    # 1. Carregar modelo
    model = xgb.Booster()
    model.load_model(model_path)
    
    # 2. Preparar dados
    features_to_drop = ['y_pred', 'y_pred_proba', 'ctm_service_route', 
                        'direction', 'lotacao_proxy_binaria', 'target']
    X = df_input.drop(columns=features_to_drop, errors='ignore')
    
    # 3. Processar datetime
    if 'event_timestamp' in X.columns:
        X['year'] = pd.to_datetime(X['event_timestamp']).dt.year
        X['month'] = pd.to_datetime(X['event_timestamp']).dt.month
        X['day'] = pd.to_datetime(X['event_timestamp']).dt.day
        X['hour'] = pd.to_datetime(X['event_timestamp']).dt.hour
        X['dayofweek'] = pd.to_datetime(X['event_timestamp']).dt.dayofweek
        X = X.drop(columns=['event_timestamp'])
    
    # 4. Encoding
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
    
    # 5. Limpar dados
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # 6. Predição
    dmatrix = xgb.DMatrix(X)
    probabilidades = model.predict(dmatrix)
    predicoes = (probabilidades >= threshold).astype(int)
    
    # 7. Retornar resultados
    resultado = df_input.copy()
    resultado['probabilidade'] = probabilidades
    resultado['predicao'] = predicoes
    resultado['confianca'] = np.where(predicoes == 1, probabilidades, 1 - probabilidades)
    
    return resultado

# Uso:
df_resultados = predict_pipeline(df_novo)
```

---

## 📚 Referências

- **XGBoost Documentation:** https://xgboost.readthedocs.io/
- **Scikit-learn Metrics:** https://scikit-learn.org/stable/modules/model_evaluation.html
- **Handling Imbalanced Data:** https://imbalanced-learn.org/

---

**Última atualização:** 28 de Outubro de 2025  
**Versão do Modelo:** 1.0 (Otimizado)
