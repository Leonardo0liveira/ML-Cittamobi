# 📚 MODELOS V5, V6 e V7 - DOCUMENTAÇÃO

## 🎯 Visão Geral

Estas três novas versões exploram **algoritmos alternativos de Gradient Boosting** e **técnicas de ensemble** para comparar com o XGBoost (V4):

| Versão | Algoritmo | Características Principais |
|--------|-----------|---------------------------|
| **V5** | LightGBM | Gradient Boosting otimizado, mais rápido que XGBoost |
| **V6** | CatBoost | Tratamento automático de categóricas, auto_class_weights |
| **V7** | Ensemble Stacking | Combina XGBoost + LightGBM + CatBoost |

---

## 🔶 V5 - LightGBM

### O que é LightGBM?
LightGBM é uma implementação de Gradient Boosting desenvolvida pela Microsoft que é:
- **Mais rápida** que XGBoost
- **Mais eficiente** em memória
- **Excelente para datasets grandes**

### Principais Configurações
```python
params = {
    'objective': 'binary',
    'num_leaves': 63,           # Árvores leaf-wise (mais eficiente)
    'max_depth': 18,
    'learning_rate': 0.015,
    'feature_fraction': 0.85,   # Subsample de features
    'bagging_fraction': 0.85,   # Subsample de linhas
    'scale_pos_weight': 12.05,  # Balanceamento
    'is_unbalance': True        # Otimização para classes desbalanceadas
}
```

### Como Executar
```bash
cd models/v5
python model_v5_lightgbm.py
```

### Arquivos Gerados
- `lightgbm_model_v5.txt` - Modelo treinado
- `visualizations/v5/` - Confusion matrix, ROC curve, feature importance
- `reports/v5_lightgbm_report.txt` - Relatório completo

### Quando Usar LightGBM
✅ Datasets grandes (>100k amostras)  
✅ Necessidade de treinamento rápido  
✅ Features numéricas predominantes  
✅ Limitações de memória  

---

## 🟢 V6 - CatBoost

### O que é CatBoost?
CatBoost é uma implementação de Gradient Boosting desenvolvida pela Yandex que é especializada em:
- **Tratamento automático de features categóricas** (sem encoding)
- **Balanceamento automático de classes**
- **Menos propenso a overfitting**
- **Ordered boosting** (previne target leakage)

### Principais Configurações
```python
model = CatBoostClassifier(
    iterations=500,
    learning_rate=0.015,
    depth=18,
    auto_class_weights='Balanced',  # 🔥 BALANCEAMENTO AUTOMÁTICO!
    l2_leaf_reg=1.0,
    subsample=0.85,
    rsm=0.85,                       # Random subspace method
    bootstrap_type='Bernoulli',
    task_type='CPU'                 # Use 'GPU' se disponível
)
```

### Como Executar
```bash
cd models/v6
python model_v6_catboost.py
```

### Arquivos Gerados
- `catboost_model_v6.cbm` - Modelo treinado
- `visualizations/v6/` - Confusion matrix, ROC curve, feature importance
- `reports/v6_catboost_report.txt` - Relatório completo

### Principais Vantagens
✅ **NÃO precisa de Label Encoding** para categóricas  
✅ **auto_class_weights='Balanced'** lida automaticamente com desbalanceamento  
✅ **Trata missing values nativamente**  
✅ **Menos hiperparâmetros** para tunar  
✅ **Reduz overfitting** naturalmente  

### Quando Usar CatBoost
✅ Muitas features categóricas (IDs, nomes, etc.)  
✅ Classes fortemente desbalanceadas  
✅ Precisa de modelo robusto com pouco tuning  
✅ Tem GPU disponível (acelera muito)  

---

## 🔷 V7 - Ensemble Stacking

### O que é Stacking?
Stacking é uma técnica de ensemble que:
1. Treina **múltiplos modelos base** (Level 0)
2. Usa as predições como features para um **meta-learner** (Level 1)
3. O meta-learner aprende **pesos ótimos** para cada modelo

### Arquitetura do V7
```
┌─────────────┐
│   XGBoost   │────┐
└─────────────┘    │
                   │    ┌──────────────────┐      ┌──────────┐
┌─────────────┐    ├───→│ Meta-Learner     │─────→│ Predição │
│  LightGBM   │────┤    │ (LogisticReg)    │      │  Final   │
└─────────────┘    │    └──────────────────┘      └──────────┘
                   │
┌─────────────┐    │
│  CatBoost   │────┘
└─────────────┘
```

### Como Funciona
1. **Treina 3 modelos** com configurações otimizadas:
   - XGBoost (V4): Advanced features + Deep trees
   - LightGBM (V5): Gradient boosting rápido
   - CatBoost (V6): Auto class weights

2. **Gera probabilidades** de cada modelo no conjunto de teste

3. **Meta-learner** (Regressão Logística) aprende:
   - Quais modelos são mais confiáveis
   - Como combinar suas predições
   - Pesos ótimos para cada modelo

### Como Executar
```bash
cd models/v7
python model_v7_stacking.py
```

### Arquivos Gerados
- `xgboost_v7.json` - Modelo base 1
- `lightgbm_v7.txt` - Modelo base 2
- `catboost_v7.cbm` - Modelo base 3
- `meta_learner_v7.pkl` - Meta-learner
- `visualizations/v7/` - Comparações e ROC curves
- `reports/v7_ensemble_report.txt` - Relatório completo

### Vantagens do Stacking
✅ **Combina pontos fortes** de cada algoritmo  
✅ **Reduz variância** - erros individuais se compensam  
✅ **Mais robusto** que modelos individuais  
✅ **Meta-learner aprende pesos automaticamente**  

### Desvantagens
❌ **Mais lento** para treinar (3 modelos + meta-learner)  
❌ **Mais complexo** para deployment  
❌ **Requer mais memória**  

### Quando Usar Stacking
✅ Máxima performance é prioridade  
✅ Tem recursos computacionais suficientes  
✅ Modelos base têm performances similares  
✅ Em competições de Machine Learning  

---

## 📊 Comparação Esperada

### Performance (estimativa baseada em literatura)

| Métrica | V4 (XGBoost) | V5 (LightGBM) | V6 (CatBoost) | V7 (Ensemble) |
|---------|--------------|---------------|---------------|---------------|
| **ROC-AUC** | 0.9731 | ~0.97-0.98 | ~0.97-0.98 | **~0.98-0.99** 🏆 |
| **F1-Macro** | 0.7760 | ~0.77-0.78 | ~0.77-0.78 | **~0.78-0.80** 🏆 |
| **Precision** | 0.59 | ~0.58-0.60 | ~0.58-0.60 | **~0.60-0.62** 🏆 |
| **Tempo Treino** | ~2-3 min | **~1-2 min** ⚡ | ~3-4 min | ~6-9 min |
| **Complexidade** | Média | Média | **Baixa** ✅ | Alta |

### Características Especiais

| Característica | V4 | V5 | V6 | V7 |
|----------------|----|----|----|----|
| Velocidade | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| Categóricas | Label Encoding | Label Encoding | **Automático** ✅ | Mixed |
| Balanceamento | scale_pos_weight | scale_pos_weight | **Auto** ✅ | Inherited |
| Overfitting | Médio | Médio | **Baixo** ✅ | Muito Baixo |
| Deployment | Fácil | Fácil | Fácil | **Complexo** |

---

## 🚀 Guia de Instalação

### Dependências Necessárias

```bash
# LightGBM
pip install lightgbm

# CatBoost
pip install catboost

# Stacking usa ambos + XGBoost (já instalado)
```

### Atualizar environment.yml

```yaml
dependencies:
  - xgboost=2.0.0
  - lightgbm=4.1.0
  - catboost=1.2.2
  - scikit-learn=1.3.0
  - pandas=2.0.0
  - numpy=1.24.0
```

---

## 📈 Como Escolher o Melhor Modelo

### Use **V5 (LightGBM)** se:
- ✅ Precisa de **velocidade** de treinamento
- ✅ Tem **dataset grande** (>500k amostras)
- ✅ Limitações de **memória**
- ✅ Features são principalmente **numéricas**

### Use **V6 (CatBoost)** se:
- ✅ Tem **muitas features categóricas** (IDs, nomes, etc.)
- ✅ Quer **menos tuning** de hiperparâmetros
- ✅ Precisa de **balanceamento automático**
- ✅ Tem **GPU disponível**

### Use **V7 (Ensemble)** se:
- ✅ **Máxima performance** é prioridade absoluta
- ✅ Tem **recursos computacionais** suficientes
- ✅ Está em uma **competição** de ML
- ✅ Modelos individuais têm ROC-AUC > 0.95

### Continue com **V4 (XGBoost)** se:
- ✅ É o **padrão da indústria** (mais adotado)
- ✅ Muita **documentação** e suporte
- ✅ **Bom equilíbrio** entre todos os aspectos
- ✅ Já está funcionando bem (0.9731 ROC-AUC)

---

## 🎯 Próximos Passos

1. **Execute os 3 modelos** e compare os resultados
2. **Analise as visualizações** em `visualizations/v5/`, `v6/`, `v7/`
3. **Compare métricas** nos relatórios gerados
4. **Escolha o melhor** baseado nas suas necessidades
5. **Documente** as conclusões no README principal

---

## 📝 Notas Importantes

- ⚠️ **CatBoost** pode ser mais lento no primeiro treinamento (compila otimizações)
- ⚠️ **Ensemble** requer 3x mais espaço em disco (salva 3 modelos)
- ⚠️ **LightGBM** pode ter resultados ligeiramente diferentes entre execuções
- ✅ Todos os modelos usam as **mesmas features do V4**
- ✅ **Threshold otimizado** para cada modelo individualmente

---

## 🔗 Referências

- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- [CatBoost Documentation](https://catboost.ai/docs/)
- [Stacking Ensemble Methods](https://scikit-learn.org/stable/modules/ensemble.html#stacking)

---

**Criado em**: Novembro 2025  
**Última atualização**: Novembro 2025
