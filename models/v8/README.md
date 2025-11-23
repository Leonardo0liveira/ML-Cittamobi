# 🚀 Cittamobi Conversion Prediction - Model V8 Production

**Status**: ✅ **PRODUÇÃO - PRONTO PARA DEPLOY**  
**Versão**: v8_production  
**Performance**: F1 Class 1 = 55.39% | ROC-AUC = 94.25%  
**Data**: 23 de Novembro de 2025

---

## 📚 Documentação Disponível

### Para Executivos e Gestores
📊 **[EXECUTIVE_SUMMARY.md](EXECUTIVE_SUMMARY.md)** - Sumário executivo com ROI e valor de negócio

### Para Desenvolvedores e ML Engineers  
📖 **[PRODUCTION_README.md](PRODUCTION_README.md)** - Documentação técnica completa do modelo  
🚀 **[DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)** - Guia passo-a-passo para deploy

---

## 🎯 Quick Start

### 1. Instalar Dependências

```bash
pip install pandas numpy scikit-learn lightgbm xgboost
```

### 2. Fazer Predições

```python
from inference_v8_production import CittamobiConversionPredictor

# Inicializar
predictor = CittamobiConversionPredictor(model_path='.')

# Predição individual
result = predictor.predict_single({
    'stop_historical_conversion': 0.35,
    'stop_density': 45.2,
    # ... outras features
})

print(f"Conversão prevista: {result['predicted_conversion']}")
print(f"Probabilidade: {result['conversion_probability']:.2%}")
```

### 3. Executar Exemplo

```bash
python inference_v8_production.py
```

---

## 📦 Artefatos

| Arquivo | Descrição | Tamanho |
|---------|-----------|---------|
| `lightgbm_model_v8_production.txt` | Modelo LightGBM | ~10 MB |
| `xgboost_model_v8_production.json` | Modelo XGBoost | ~15 MB |
| `scaler_v8_production.pkl` | Normalizador | ~50 KB |
| `selected_features_v8_production.txt` | Lista de features | ~2 KB |
| `model_config_v8_production.json` | Configuração | ~5 KB |

---

## 📊 Performance

### Métricas Principais

```
F1 Score Classe 1 (Conversão):     0.5539  (55.39%)
F1 Score Classe 0 (Não-Conversão): 0.9576  (95.76%)
ROC-AUC:                            0.9425  (94.25%)
Accuracy:                           0.9240  (92.40%)
Precision Classe 1:                 0.6474  (64.74%)
Recall Classe 1:                    0.4848  (48.48%)
```

### Confusion Matrix

```
                 Predicted
                 0        1
Actual    0   [54,060]  [1,428]
          1   [ 3,522]  [3,100]
```

---

## 🏗️ Arquitetura

### Ensemble Otimizado
- **LightGBM**: 48.5% do peso
- **XGBoost**: 51.5% do peso

### Features (45 total)
- **6 Geographic Features**: localização, densidade, distância CBD
- **10 Dynamic Features**: temporal, usuário, interações
- **29 Base Features**: features do dataset original

### Técnicas Avançadas
- ✅ Threshold dinâmico adaptativo
- ✅ Sample weights dinâmicos
- ✅ Normalização StandardScaler
- ✅ Validação estratificada

---

## 🔄 Histórico de Versões

### v8_production (23/11/2025) - **ATUAL**
- ✅ Baseado na Fase 2A (melhor performance)
- ✅ F1 Classe 1: 0.5539 (+32% vs v7)
- ✅ Ensemble LightGBM + XGBoost
- ✅ 16 features customizadas
- ✅ Threshold dinâmico
- ✅ Documentação completa

### v8_phase2b (23/11/2025) - DESCARTADO
- ❌ F1 Classe 1: 0.4871 (-12% vs 2A)
- ❌ SMOTE prejudicou performance
- ❌ Features temporais sem valor
- 📝 Lições aprendidas documentadas

### v8_phase2a (22/11/2025)
- ✅ F1 Classe 1: 0.5539 (baseline)
- ✅ Threshold dinâmico implementado
- ✅ Sample weights implementados

### v7 (20/11/2025)
- F1 Classe 1: 0.42
- Feature selection implementada

---

## 🚀 Como Usar

### Cenário 1: API REST

```python
from flask import Flask, request, jsonify
from inference_v8_production import CittamobiConversionPredictor

app = Flask(__name__)
predictor = CittamobiConversionPredictor()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    result = predictor.predict_single(data)
    return jsonify(result)

app.run(port=5000)
```

### Cenário 2: Batch Processing

```python
import pandas as pd
from inference_v8_production import CittamobiConversionPredictor

predictor = CittamobiConversionPredictor()

# Carregar dados
df = pd.read_csv('eventos.csv')

# Predições
predictions = predictor.predict(df)

# Salvar
df['predicted_conversion'] = predictions
df.to_csv('eventos_com_predicoes.csv')
```

### Cenário 3: Real-time Streaming

```python
from inference_v8_production import CittamobiConversionPredictor

predictor = CittamobiConversionPredictor()

# Exemplo com Kafka
from kafka import KafkaConsumer

consumer = KafkaConsumer('events-topic')

for message in consumer:
    event = json.loads(message.value)
    result = predictor.predict_single(event)
    
    if result['predicted_conversion']:
        # Ação: enviar notificação, etc
        send_notification(event['user_id'])
```

---

## 📈 Monitoramento

### Métricas a Acompanhar

```python
from sklearn.metrics import f1_score, roc_auc_score

# Performance
f1 = f1_score(y_true, y_pred, pos_label=1)
auc = roc_auc_score(y_true, y_proba)

# Alertas
if f1 < 0.50:
    send_alert("F1 Score abaixo do esperado!")
if auc < 0.90:
    send_alert("ROC-AUC degradou!")
```

### Dashboard Recomendado

- F1 Score por dia
- ROC-AUC por semana
- Distribuição de probabilidades
- Taxa de conversão real vs prevista
- Latência de inferência

---

## 🔧 Manutenção

### Retreinamento

Execute quando:
- F1 < 0.50 por 3 dias consecutivos
- A cada 3-6 meses (periodicidade)
- 100K+ novos eventos rotulados

```bash
# Retreinar
python model_v8_production.py

# Validar
python inference_v8_production.py

# Deploy se melhor que atual
```

---

## ⚠️ Troubleshooting

### Erro: "Features faltando"

```python
# Ver features necessárias
with open('selected_features_v8_production.txt', 'r') as f:
    features = [line.strip() for line in f]
print(features)
```

### Erro: "Modelo não carrega"

```bash
# Verificar versões
pip list | grep -E "lightgbm|xgboost|scikit-learn"

# Versões mínimas:
# lightgbm>=3.3.0
# xgboost>=1.7.0
# scikit-learn>=1.2.0
```

---

## 📞 Suporte

- **Documentação Técnica**: [PRODUCTION_README.md](PRODUCTION_README.md)
- **Guia de Deploy**: [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)
- **Sumário Executivo**: [EXECUTIVE_SUMMARY.md](EXECUTIVE_SUMMARY.md)
- **Desenvolvedor**: Stefano (IBMEC)

---

## 📄 Licença

Propriedade do cliente Cittamobi. Uso restrito interno.

---

## ✨ Status do Projeto

```
✅ Desenvolvimento:      CONCLUÍDO
✅ Treinamento:          CONCLUÍDO  
✅ Validação:            CONCLUÍDA
✅ Documentação:         COMPLETA
⏳ Deploy Staging:       PENDENTE
⏳ Deploy Produção:      PENDENTE
⏳ Monitoramento:        PENDENTE
```

---

**🎉 Modelo V8 Production - Pronto para transformar dados em valor! 🎉**

*Última atualização: 23 de Novembro de 2025*
