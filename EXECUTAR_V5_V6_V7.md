# 🚀 GUIA RÁPIDO - EXECUTAR V5, V6 E V7

## 📦 Instalação Rápida

```bash
# 1. Instalar dependências
pip install lightgbm catboost

# Ou usar o arquivo de requirements
pip install -r requirements_v5_v6_v7.txt
```

---

## ▶️ Execução dos Modelos

### V5 - LightGBM (Rápido) ⚡
```bash
cd models/v5
python model_v5_lightgbm.py
```
**Tempo estimado**: ~2-3 minutos  
**Output**: `lightgbm_model_v5.txt` + visualizações

---

### V6 - CatBoost (Categóricas) 🟢
```bash
cd models/v6
python model_v6_catboost.py
```
**Tempo estimado**: ~3-4 minutos  
**Output**: `catboost_model_v6.cbm` + visualizações

---

### V7 - Ensemble Stacking (Melhor) 🏆
```bash
cd models/v7
python model_v7_stacking.py
```
**Tempo estimado**: ~8-10 minutos (treina 3 modelos)  
**Output**: 4 arquivos (xgb, lgb, cat, meta-learner) + visualizações

---

## 📊 O que cada modelo gera

### Arquivos de Modelo
- **V5**: `lightgbm_model_v5.txt`
- **V6**: `catboost_model_v6.cbm`
- **V7**: `xgboost_v7.json`, `lightgbm_v7.txt`, `catboost_v7.cbm`, `meta_learner_v7.pkl`

### Visualizações (em `visualizations/vX/`)
- `confusion_matrix_vX.png` - Matriz de confusão
- `roc_curve_vX.png` - Curva ROC
- `feature_importance_vX.png` - Importância das features
- `learning_curves_vX.png` - Curvas de aprendizado (V5)
- `roc_curves_comparison_v7.png` - Comparação de todos (V7)
- `models_comparison_v7.png` - Gráfico de barras comparativo (V7)

### Relatórios (em `reports/`)
- `v5_lightgbm_report.txt`
- `v6_catboost_report.txt`
- `v7_ensemble_report.txt`

---

## 🔍 Como Analisar os Resultados

### 1. Verificar Métricas no Console
Durante a execução, cada modelo imprime:
```
📊 MÉTRICAS FINAIS:
   ROC-AUC:      0.XXXX
   Accuracy:     0.XXXX
   Precision:    0.XXXX
   Recall:       0.XXXX
   F1-Score:     0.XXXX
   F1-Macro:     0.XXXX
```

### 2. Analisar Visualizações
Abra os arquivos em `visualizations/v5/`, `v6/`, `v7/`:
- **Confusion Matrix**: Ver quantos FP/FN/TP/TN
- **ROC Curve**: Curva de desempenho (quanto mais próximo de 1.0, melhor)
- **Feature Importance**: Quais features são mais importantes

### 3. Ler Relatórios Completos
Arquivos `.txt` em `reports/` contêm todas as métricas e top features.

---

## 📈 Comparação de Performance

Após executar os 3 modelos, compare:

| Métrica | V4 (XGBoost) | V5 (LightGBM) | V6 (CatBoost) | V7 (Ensemble) |
|---------|--------------|---------------|---------------|---------------|
| **ROC-AUC** | 0.9731 | ? | ? | ? |
| **F1-Macro** | 0.7760 | ? | ? | ? |
| **Precision** | 0.59 | ? | ? | ? |
| **Recall** | 0.55 | ? | ? | ? |
| **Tempo** | ~3 min | ~2 min | ~4 min | ~10 min |

---

## 🎯 Qual Modelo Escolher?

### Use **V5 (LightGBM)** se:
✅ Precisa de velocidade  
✅ Dataset grande (>500k amostras)  
✅ Limitações de memória  

### Use **V6 (CatBoost)** se:
✅ Muitas features categóricas (IDs, nomes)  
✅ Quer menos tuning de hiperparâmetros  
✅ Tem GPU disponível  

### Use **V7 (Ensemble)** se:
✅ Máxima performance é prioridade  
✅ Tem recursos computacionais  
✅ Em competições de ML  

### Continue com **V4 (XGBoost)** se:
✅ É o padrão da indústria  
✅ Já está funcionando bem (0.9731 ROC-AUC)  
✅ Boa documentação e suporte  

---

## ⚠️ Troubleshooting

### Erro: "ModuleNotFoundError: No module named 'lightgbm'"
```bash
pip install lightgbm
```

### Erro: "ModuleNotFoundError: No module named 'catboost'"
```bash
pip install catboost
```

### Erro de memória (MemoryError)
- Reduza o `LIMIT` na query SQL (linha 23-26 de cada arquivo)
- Exemplo: `LIMIT 100000` em vez de `LIMIT 200000`

### Modelo V7 muito lento
- É esperado! Ele treina 3 modelos + meta-learner
- Para acelerar: use menos dados ou pule o V7

### Resultados diferentes a cada execução
- Normal para LightGBM (pequenas variações)
- Use `random_seed=42` para reprodutibilidade

---

## 📝 Checklist de Execução

- [ ] Instalar dependências (`pip install lightgbm catboost`)
- [ ] Executar V5 - LightGBM
- [ ] Executar V6 - CatBoost
- [ ] Executar V7 - Ensemble Stacking
- [ ] Analisar visualizações em `visualizations/v5/`, `v6/`, `v7/`
- [ ] Comparar métricas nos relatórios
- [ ] Escolher melhor modelo para seu caso de uso
- [ ] Documentar resultados no README principal

---

## 🔗 Links Úteis

- [Documentação Completa V5-V7](../docs/V5_V6_V7_EXPLICACAO.md)
- [LightGBM Docs](https://lightgbm.readthedocs.io/)
- [CatBoost Docs](https://catboost.ai/docs/)
- [Stacking Ensemble](https://scikit-learn.org/stable/modules/ensemble.html#stacking)

---

**Última atualização**: Novembro 2025
