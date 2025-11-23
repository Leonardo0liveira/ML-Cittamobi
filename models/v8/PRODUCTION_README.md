# 🚀 Model V8 - Versão de Produção

## 📋 Informações Gerais

**Versão**: v8_production  
**Data de Criação**: 23 de Novembro de 2025  
**Objetivo**: Predição de conversão de usuários em pontos de ônibus

---

## 🏆 Performance do Modelo

### Métricas Principais

| Métrica | Valor | Descrição |
|---------|-------|-----------|
| **F1 Classe 1 (Conversão)** | **0.5539** | Equilíbrio entre precisão e recall para conversões |
| **F1 Classe 0 (Não-Conversão)** | **0.9576** | Equilíbrio entre precisão e recall para não-conversões |
| **ROC-AUC** | **0.9425** | Capacidade de discriminação do modelo |
| **F1-Macro** | **0.7558** | Média das métricas F1 das duas classes |
| **Accuracy** | **0.9240** | Taxa de acertos geral |

### Confusion Matrix

```
                    Predicted
                    0        1
Actual    0     [TN]      [FP]
          1     [FN]      [TP]
```

- **True Negatives (TN)**: ~54,000 (não-conversões corretamente identificadas)
- **True Positives (TP)**: ~3,100 (conversões corretamente identificadas)
- **False Positives (FP)**: ~1,400 (falsos alarmes)
- **False Negatives (FN)**: ~3,500 (conversões perdidas)

---

## 🔧 Arquitetura do Modelo

### Ensemble de Modelos

O modelo final é um **ensemble otimizado** de dois algoritmos:

1. **LightGBM** (48.5%)
   - Gradient Boosting Decision Tree
   - 300 árvores
   - Learning rate: 0.05
   - 63 folhas por árvore

2. **XGBoost** (51.5%)
   - Extreme Gradient Boosting
   - 300 árvores
   - Learning rate: 0.05
   - Profundidade máxima: 8

### Features Engineered (16 features customizadas)

#### Geographic Features (6 features)
1. **stop_historical_conversion**: Taxa média de conversão por parada
2. **stop_density**: Densidade de paradas (inverso da distância média aos vizinhos)
3. **dist_to_nearest_cbd**: Distância ao CBD mais próximo (SP, RJ, BH, Curitiba, POA)
4. **stop_cluster**: Cluster DBSCAN da parada
5. **cluster_conversion_rate**: Taxa de conversão do cluster
6. **stop_volatility**: Volatilidade de conversões na parada

#### Dynamic Features (10 features)
1. **hour_conversion_rate**: Taxa de conversão por hora do dia
2. **dow_conversion_rate**: Taxa de conversão por dia da semana
3. **stop_hour_conversion**: Taxa de conversão parada × hora
4. **geo_temporal**: Distância CBD × hora de pico
5. **density_peak**: Densidade × hora de pico
6. **user_conversion_rate**: Taxa de conversão por usuário
7. **user_vs_stop_ratio**: Razão paradas únicas / eventos por usuário
8. **stop_rarity**: Raridade da parada (inverso da frequência)
9. **user_rarity**: Raridade do usuário (inverso da frequência)
10. **stop_dist_std**: Desvio padrão de distâncias na parada

---

## 🎯 Estratégia de Threshold Dinâmico

O modelo utiliza **thresholds adaptativos** baseados na taxa de conversão histórica da parada:

| Taxa de Conversão Histórica | Threshold | Estratégia |
|----------------------------|-----------|------------|
| ≥ 50% (Alta) | **0.40** | Mais agressivo - capturar mais conversões |
| 30-50% (Média) | **0.50** | Balanceado |
| 10-30% (Baixa) | **0.60** | Mais conservador |
| < 10% (Muito Baixa) | **0.75** | Muito conservador - evitar falsos positivos |

### Distribuição de Thresholds no Dataset

- **0.40**: ~17% das amostras (paradas de alta conversão)
- **0.50**: ~4% das amostras (paradas de média-alta conversão)
- **0.60**: ~4% das amostras (paradas de média-baixa conversão)
- **0.75**: ~75% das amostras (paradas de baixa conversão)

---

## 📦 Artefatos de Produção

### Arquivos Salvos

1. **lightgbm_model_v8_production.txt**
   - Modelo LightGBM serializado
   - Formato: LightGBM nativo

2. **xgboost_model_v8_production.json**
   - Modelo XGBoost serializado
   - Formato: JSON (compatível com qualquer linguagem)

3. **scaler_v8_production.pkl**
   - StandardScaler do scikit-learn
   - Necessário para normalizar features

4. **selected_features_v8_production.txt**
   - Lista de features utilizadas (45 features)
   - Ordem deve ser preservada na inferência

5. **model_config_v8_production.json**
   - Configuração completa do modelo
   - Pesos do ensemble
   - Regras de threshold
   - Parâmetros de treinamento
   - Métricas de performance

---

## 🔨 Como Usar o Modelo

### 1. Carregar o Modelo

```python
import lightgbm as lgb
import xgboost as xgb
import pickle
import json
import pandas as pd
import numpy as np

# Carregar modelos
lgb_model = lgb.Booster(model_file='lightgbm_model_v8_production.txt')
xgb_model = xgb.Booster()
xgb_model.load_model('xgboost_model_v8_production.json')

# Carregar scaler
with open('scaler_v8_production.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Carregar configuração
with open('model_config_v8_production.json', 'r') as f:
    config = json.load(f)

# Carregar lista de features
with open('selected_features_v8_production.txt', 'r') as f:
    feature_cols = [line.strip() for line in f]
```

### 2. Preparar os Dados

```python
# Exemplo de dados de entrada
# df deve conter todas as features base necessárias

# Feature Engineering (implementar as 16 features customizadas)
# ... (ver código de feature engineering no script de treinamento)

# Selecionar features
X = df[feature_cols].copy()

# Normalizar
X_scaled = scaler.transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=feature_cols)
```

### 3. Fazer Predições

```python
# Predições dos modelos individuais
pred_lgb = lgb_model.predict(X_scaled)
pred_xgb = xgb_model.predict(xgb.DMatrix(X_scaled))

# Ensemble
w_lgb = config['ensemble_weights']['lightgbm']
w_xgb = config['ensemble_weights']['xgboost']
pred_ensemble = w_lgb * pred_lgb + w_xgb * pred_xgb

# Aplicar threshold dinâmico
def get_dynamic_threshold(stop_conv):
    rules = config['threshold_rules']
    if stop_conv >= rules['high_conversion']['min']:
        return rules['high_conversion']['threshold']
    elif stop_conv >= rules['medium_conversion']['min']:
        return rules['medium_conversion']['threshold']
    elif stop_conv >= rules['low_conversion']['min']:
        return rules['low_conversion']['threshold']
    else:
        return rules['very_low_conversion']['threshold']

thresholds = df['stop_historical_conversion'].apply(get_dynamic_threshold)
predictions = (pred_ensemble > thresholds).astype(int)
```

### 4. Interpretar Resultados

```python
# Adicionar probabilidades e predições ao DataFrame
df['conversion_probability'] = pred_ensemble
df['predicted_conversion'] = predictions
df['threshold_used'] = thresholds

# Exemplo de uso
high_conversion = df[df['predicted_conversion'] == 1]
print(f"Conversões previstas: {len(high_conversion)}")
print(f"Probabilidade média: {high_conversion['conversion_probability'].mean():.2%}")
```

---

## 📊 Casos de Uso

### 1. Predição em Tempo Real
- Receber evento de usuário em parada
- Calcular features em tempo real
- Executar modelo
- Retornar probabilidade de conversão

### 2. Predição em Batch
- Processar lote de eventos históricos
- Gerar predições para análise
- Identificar padrões de conversão

### 3. Otimização de Rotas
- Identificar paradas de alta conversão
- Priorizar rotas com maior potencial
- Alocar recursos de marketing

### 4. Análise de Performance
- Monitorar taxa de conversão por parada
- Identificar anomalias
- Ajustar estratégias de negócio

---

## ⚠️ Requisitos e Dependências

### Python Packages
```
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
lightgbm>=3.3.0
xgboost>=1.7.0
google-cloud-bigquery>=3.0.0
```

### Hardware Recomendado
- **CPU**: 4+ cores
- **RAM**: 8GB+ (16GB recomendado)
- **Disco**: 1GB para artefatos

### Tempo de Inferência
- **Predição individual**: ~5ms
- **Batch (1000 eventos)**: ~1s
- **Batch (100,000 eventos)**: ~30s

---

## 🔄 Manutenção e Retreinamento

### Quando Retreinar?

1. **Performance Degradation**: F1 Classe 1 cai abaixo de 0.50
2. **Data Drift**: Distribuição de features muda significativamente
3. **Periodicidade**: A cada 3-6 meses
4. **Novos Dados**: Quando acumular 100K+ novos eventos rotulados

### Monitoramento Contínuo

Monitore as seguintes métricas em produção:

- **F1 Score Classe 1**: Deve permanecer ≥ 0.50
- **ROC-AUC**: Deve permanecer ≥ 0.90
- **Distribuição de Thresholds**: Verificar se padrões mudam
- **Calibração**: Verificar se probabilidades permanecem calibradas

---

## 📝 Changelog

### v8_production (23/11/2025)
- ✅ Implementação inicial baseada em Fase 2A
- ✅ Ensemble LightGBM + XGBoost otimizado
- ✅ 16 features customizadas (6 geographic + 10 dynamic)
- ✅ Threshold dinâmico adaptativo
- ✅ Sample weights dinâmicos
- ✅ F1 Classe 1: 0.5539 (55.39%)
- ✅ ROC-AUC: 0.9425 (94.25%)

---

## 👥 Contato e Suporte

Para dúvidas, problemas ou sugestões de melhorias:

- **Desenvolvedor**: Stefano
- **Projeto**: Cittamobi Forecast - IBMEC
- **Data**: Novembro 2025

---

## 📄 Licença

Este modelo é propriedade do cliente e destinado exclusivamente para uso interno.

---

**✨ Modelo pronto para deploy em produção! ✨**
