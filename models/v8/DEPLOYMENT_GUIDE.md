# 🚀 Guia Rápido de Deploy - Model V8 Production

## ✅ Checklist Pré-Deploy

- [ ] Todos os artefatos estão disponíveis
- [ ] Ambiente Python configurado (Python 3.8+)
- [ ] Dependências instaladas
- [ ] Teste de inferência executado com sucesso
- [ ] Performance validada (F1 Class 1 ≥ 0.50)
- [ ] Documentação revisada

---

## 📦 1. Artefatos Necessários

Certifique-se de ter os seguintes arquivos:

```
models/v8/
├── lightgbm_model_v8_production.txt      # Modelo LightGBM
├── xgboost_model_v8_production.json      # Modelo XGBoost
├── scaler_v8_production.pkl              # Normalizador
├── selected_features_v8_production.txt   # Lista de features
├── model_config_v8_production.json       # Configuração
├── inference_v8_production.py            # Script de inferência
├── PRODUCTION_README.md                  # Documentação completa
└── DEPLOYMENT_GUIDE.md                   # Este guia
```

**Tamanho total**: ~50MB

---

## 🔧 2. Instalação do Ambiente

### Opção A: Conda (Recomendado)

```bash
# Criar ambiente
conda create -n cittamobi-prod python=3.10

# Ativar ambiente
conda activate cittamobi-prod

# Instalar dependências
pip install pandas numpy scikit-learn lightgbm xgboost google-cloud-bigquery
```

### Opção B: venv

```bash
# Criar ambiente virtual
python -m venv venv-cittamobi

# Ativar ambiente (Linux/Mac)
source venv-cittamobi/bin/activate

# Ativar ambiente (Windows)
venv-cittamobi\Scripts\activate

# Instalar dependências
pip install pandas numpy scikit-learn lightgbm xgboost google-cloud-bigquery
```

### Opção C: Requirements File

```bash
# Criar arquivo requirements.txt
cat > requirements.txt << EOF
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
lightgbm>=3.3.0
xgboost>=1.7.0
google-cloud-bigquery>=3.0.0
EOF

# Instalar
pip install -r requirements.txt
```

---

## 🧪 3. Teste de Validação

### Teste 1: Verificar Artefatos

```bash
# Verificar se todos os arquivos existem
ls -lh lightgbm_model_v8_production.txt
ls -lh xgboost_model_v8_production.json
ls -lh scaler_v8_production.pkl
ls -lh selected_features_v8_production.txt
ls -lh model_config_v8_production.json
```

### Teste 2: Carregar Modelos

```python
import lightgbm as lgb
import xgboost as xgb
import pickle
import json

# Teste LightGBM
lgb_model = lgb.Booster(model_file='lightgbm_model_v8_production.txt')
print("✓ LightGBM carregado")

# Teste XGBoost
xgb_model = xgb.Booster()
xgb_model.load_model('xgboost_model_v8_production.json')
print("✓ XGBoost carregado")

# Teste Scaler
with open('scaler_v8_production.pkl', 'rb') as f:
    scaler = pickle.load(f)
print("✓ Scaler carregado")

# Teste Config
with open('model_config_v8_production.json', 'r') as f:
    config = json.load(f)
print(f"✓ Config carregado - F1: {config['metrics']['f1_class_1']:.4f}")
```

### Teste 3: Executar Script de Inferência

```bash
python inference_v8_production.py
```

**Saída esperada**:
- Modelos carregados com sucesso
- Exemplos de predição executados
- Nenhum erro

---

## 🌐 4. Integração com Aplicação

### Opção A: API REST (Flask)

```python
from flask import Flask, request, jsonify
from inference_v8_production import CittamobiConversionPredictor

app = Flask(__name__)
predictor = CittamobiConversionPredictor(model_path='./models/v8/')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Receber dados
        data = request.json
        
        # Fazer predição
        result = predictor.predict_single(data)
        
        return jsonify({
            'success': True,
            'prediction': result
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

**Testar API**:
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"stop_historical_conversion": 0.35, ...}'
```

### Opção B: API REST (FastAPI)

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from inference_v8_production import CittamobiConversionPredictor

app = FastAPI()
predictor = CittamobiConversionPredictor(model_path='./models/v8/')

class PredictionRequest(BaseModel):
    stop_historical_conversion: float
    stop_density: float
    dist_to_nearest_cbd: float
    # ... outras features

@app.post("/predict")
async def predict(request: PredictionRequest):
    try:
        result = predictor.predict_single(request.dict())
        return {"success": True, "prediction": result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Executar: uvicorn api:app --host 0.0.0.0 --port 5000
```

### Opção C: Batch Processing

```python
from inference_v8_production import CittamobiConversionPredictor
import pandas as pd

# Inicializar preditor
predictor = CittamobiConversionPredictor(model_path='./models/v8/')

# Carregar dados
df = pd.read_csv('eventos_para_processar.csv')

# Fazer predições
predictions, probas, thresholds = predictor.predict(df, return_proba=True)

# Adicionar ao DataFrame
df['predicted_conversion'] = predictions
df['conversion_probability'] = probas
df['threshold_used'] = thresholds

# Salvar resultados
df.to_csv('eventos_com_predicoes.csv', index=False)
print(f"✓ {len(df)} eventos processados")
```

---

## 📊 5. Monitoramento em Produção

### Métricas a Monitorar

1. **Performance Metrics**
   - F1 Score Classe 1 (target: ≥ 0.50)
   - ROC-AUC (target: ≥ 0.90)
   - Accuracy (target: ≥ 0.85)

2. **Data Quality**
   - Valores missing por feature
   - Distribuição de features (detectar data drift)
   - Outliers

3. **Operational Metrics**
   - Latência média de predição
   - Throughput (predições/segundo)
   - Taxa de erro

### Script de Monitoramento

```python
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
from datetime import datetime

def monitor_model_performance(y_true, y_pred, y_proba):
    """
    Monitora performance do modelo em produção.
    """
    metrics = {
        'timestamp': datetime.now().isoformat(),
        'f1_class_1': f1_score(y_true, y_pred, pos_label=1),
        'roc_auc': roc_auc_score(y_true, y_proba),
        'n_samples': len(y_true),
        'conversion_rate': y_true.mean()
    }
    
    # Alertas
    if metrics['f1_class_1'] < 0.50:
        print("⚠️  ALERTA: F1 Classe 1 abaixo de 0.50!")
    if metrics['roc_auc'] < 0.90:
        print("⚠️  ALERTA: ROC-AUC abaixo de 0.90!")
    
    return metrics

# Usar periodicamente
metrics = monitor_model_performance(y_true, y_pred, y_proba)
print(f"F1 Classe 1: {metrics['f1_class_1']:.4f}")
```

---

## 🔄 6. Retreinamento

### Quando Retreinar?

Retreine o modelo quando:
1. **F1 Classe 1 < 0.50** por 3 dias consecutivos
2. **ROC-AUC < 0.90** por 1 semana
3. **Data Drift** detectado (PSI > 0.25)
4. **Periodicidade**: A cada 3-6 meses
5. **Novos dados**: 100K+ eventos rotulados acumulados

### Processo de Retreinamento

```bash
# 1. Coletar novos dados
# 2. Executar script de treinamento
python model_v8_production.py

# 3. Validar novo modelo
python inference_v8_production.py

# 4. Comparar métricas (novo vs atual)
# 5. Se melhor, fazer deploy do novo modelo
# 6. Manter backup do modelo anterior
```

---

## 🐳 7. Deploy com Docker (Opcional)

### Dockerfile

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Instalar dependências
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar artefatos do modelo
COPY models/v8/ ./models/v8/

# Copiar API
COPY api.py .

# Expor porta
EXPOSE 5000

# Comando de inicialização
CMD ["python", "api.py"]
```

### Build e Run

```bash
# Build
docker build -t cittamobi-predictor:v8 .

# Run
docker run -p 5000:5000 cittamobi-predictor:v8

# Testar
curl http://localhost:5000/predict -X POST -d '{"stop_historical_conversion": 0.35}'
```

---

## ☁️ 8. Deploy na Cloud (Exemplos)

### Google Cloud Run

```bash
# Fazer push da imagem
gcloud builds submit --tag gcr.io/[PROJECT-ID]/cittamobi-predictor:v8

# Deploy
gcloud run deploy cittamobi-predictor \
  --image gcr.io/[PROJECT-ID]/cittamobi-predictor:v8 \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

### AWS Lambda (com serverless)

```yaml
# serverless.yml
service: cittamobi-predictor

provider:
  name: aws
  runtime: python3.10
  region: us-east-1

functions:
  predict:
    handler: handler.predict
    memorySize: 2048
    timeout: 30
    events:
      - http:
          path: predict
          method: post
```

### Azure Functions

```bash
# Criar função
func init CittamobiPredictor --python

# Deploy
func azure functionapp publish cittamobi-predictor-app
```

---

## 🔐 9. Segurança

### Checklist de Segurança

- [ ] API protegida com autenticação (OAuth2/JWT)
- [ ] Rate limiting implementado
- [ ] Input validation em todas as requisições
- [ ] Logs de acesso configurados
- [ ] Dados sensíveis mascarados nos logs
- [ ] HTTPS obrigatório
- [ ] Backup dos modelos em storage seguro

### Exemplo de Autenticação (FastAPI)

```python
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != "SECRET_TOKEN":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )
    return credentials.credentials

@app.post("/predict")
async def predict(request: PredictionRequest, token: str = Depends(verify_token)):
    # ... predição
```

---

## 📝 10. Troubleshooting

### Problema: Erro ao carregar modelo

**Solução**:
```python
# Verificar versões
import lightgbm
import xgboost
print(f"LightGBM: {lightgbm.__version__}")
print(f"XGBoost: {xgboost.__version__}")

# Versões recomendadas:
# LightGBM >= 3.3.0
# XGBoost >= 1.7.0
```

### Problema: Features faltando

**Solução**:
```python
# Verificar features necessárias
with open('selected_features_v8_production.txt', 'r') as f:
    required_features = [line.strip() for line in f]

# Verificar features presentes
missing = set(required_features) - set(df.columns)
print(f"Features faltando: {missing}")
```

### Problema: Performance degradada

**Solução**:
1. Verificar data drift
2. Coletar mais dados rotulados
3. Retreinar modelo
4. Ajustar thresholds dinâmicos

---

## ✅ Checklist Final de Deploy

- [ ] Ambiente configurado
- [ ] Dependências instaladas
- [ ] Modelos carregando corretamente
- [ ] Teste de inferência passou
- [ ] API funcionando
- [ ] Monitoramento configurado
- [ ] Documentação revisada
- [ ] Backup dos artefatos criado
- [ ] Equipe treinada
- [ ] Plano de rollback definido

---

## 📞 Suporte

Para problemas ou dúvidas:
1. Consultar `PRODUCTION_README.md`
2. Verificar logs de erro
3. Executar testes de validação
4. Contatar desenvolvedor: Stefano (IBMEC)

---

**🎉 Deploy pronto para produção! Boa sorte! 🎉**
