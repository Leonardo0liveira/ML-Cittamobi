# 🚍 Projeto Machine Learning - Cittamobi Forecast

## 📋 Visão Geral

Projeto de previsão de conversão de eventos de usuários de transporte público utilizando múltiplos algoritmos de Machine Learning com foco em otimização e ensemble methods.

**Objetivo**: Prever se um usuário irá converter (realizar uma ação desejada) com base em seus padrões de uso e características dos eventos.

---

## 📊 Resultados - Modelos Otimizados

| Modelo | ROC-AUC | F1-Score | F1-Macro | Accuracy | Precision | Recall | Status |
|--------|---------|----------|----------|----------|-----------|--------|--------|
| **Random Forest** | **80.83%** | **40.85%** | **66.48%** | **86.41%** | **33.71%** | **52.31%** | 🏆 **Melhor** |
| **CatBoost** | **80.40%** | **38.48%** | **66.48%** | **83.98%** | **28.38%** | **61.66%** | 🥈 2º lugar |
| **Stacking Ensemble** | **80.48%** | **39.75%** | **66.37%** | **86.06%** | **32.27%** | **52.38%** | 🏅 3º lugar |
| SVM | 79.92% | 40.31% | 66.21% | 86.24% | 33.71% | 50.55% | ✅ |
| SGD (Hinge) | 77.47% | 37.83% | 64.55% | 85.46% | 34.94% | 42.43% | ✅ |
| Naive Bayes | 77.67% | 28.89% | 61.65% | 70.50% | 17.74% | 78.92% | ✅ |
| KNN (K=51) | 75.33% | 39.61% | 64.83% | 84.80% | 40.33% | 38.93% | ✅ |
| Decision Tree | 75.02% | 34.38% | 62.84% | 82.56% | 27.55% | 46.64% | ✅ |

**Técnicas Aplicadas**: Undersampling (2:1), TimeSeriesSplit CV (5 folds), StandardScaler normalization

---

## 📁 Estrutura do Projeto (Cookie Cutter Data Science)

```
Projeto Machine Learning/
├── README.md                      # Documentação principal
├── environment.yml               # Ambiente conda
├── INDEX.md                      # Índice de documentação
├── ORGANIZACAO.md               # Detalhes da organização
│
├── data/                        # Dados em diferentes estágios
│   ├── 01_raw/                 # Dados brutos do BigQuery
│   ├── 02_interim/             # Dados com expanding windows
│   └── 03_processed/           # Dados prontos para modelagem
│
├── models/                      # Código dos modelos
│   ├── SGDClassifier/          # SGD com 3 loss functions
│   ├── KNN/                    # K-Nearest Neighbors
│   ├── DecisionTrees/          # Decision Tree com múltiplas profundidades
│   ├── NaiveBayes/             # Gaussian, Multinomial, Bernoulli
│   ├── SVM/                    # Support Vector Machine
│   ├── RandomForest/           # Random Forest (melhor modelo)
│   ├── catboost/               # CatBoost gradient boosting
│   ├── lightgbm/               # LightGBM (experimental)
│   ├── stacking_ensemble.py    # Ensemble RF+CB+SGD
│   ├── all_models_comparison.py # Comparação unificada
│   ├── trained/                # Modelos salvos (.pkl, .json)
│   ├── predictions/            # Predições dos modelos
│   └── archive/                # Versões antigas (v1-v4)
│
├── notebooks/                   # Notebooks e scripts de análise
│   ├── exploratory/            # Análise exploratória, testes
│   └── final/                  # Notebooks finalizados
│
├── reports/                     # Resultados e análises
│   ├── figures/                # Visualizações (ROC, confusion matrix, etc.)
│   └── model_evaluations/      # Relatórios .txt, .csv, .md
│
├── src/                        # Código fonte reutilizável
│   ├── __init__.py
│   ├── data/                   # Scripts de carregamento de dados
│   │   └── __init__.py
│   ├── features/               # Engenharia de features
│   │   └── __init__.py
│   └── models/                 # Utilitários de modelagem
│       └── __init__.py
│
└── docs/                       # Documentação detalhada
    ├── ANALISE_RESULTADOS.md
    ├── COMPARACAO_V1_V2.md
    ├── GUIA_DE_USO.md
    └── V3_ENHANCED_EXPLICACAO.md

---

## 🚀 Como Usar

### 1. Configurar Ambiente

```bash
conda env create -f environment.yml
conda activate cittamobi-forecast
```

### 2. Executar Modelos

#### Modelos Individuais
```bash
# Random Forest (melhor modelo - 80.83% ROC-AUC)
cd models/RandomForest
python random_forest_optimized.py

# CatBoost (segundo melhor - 80.40% ROC-AUC)
cd models/catboost
python catboost_optimized.py

# Outros modelos
cd models/SGDClassifier && python sgd_optimized.py
cd models/SVM && python svm_optimized.py
cd models/KNN && python knn_optimized.py
cd models/NaiveBayes && python nb_optimized.py
cd models/DecisionTrees && python decision_tree_optimized.py
```

#### Ensemble Stacking (80.48% ROC-AUC)
```bash
python models/stacking_ensemble.py
```

#### Comparação de Todos os Modelos
```bash
python models/all_models_comparison.py
```

### 3. Ver Resultados

- **Visualizações**: `reports/figures/` (ROC curves, confusion matrices, feature importance)
- **Relatórios**: `reports/model_evaluations/` (classification reports .txt, results .csv)
- **Comparações**: `reports/model_evaluations/modelos_otimizados_comparacao.md`
- **Modelos Salvos**: `models/*/xgboost_model_*.json` ou `models/trained/`

---

## 📚 Documentação

### Guias de Uso
- **[GUIA_DE_USO.md](docs/GUIA_DE_USO.md)**: Como executar e interpretar os modelos
- **[INDEX.md](INDEX.md)**: Índice completo da documentação
- **[ORGANIZACAO.md](ORGANIZACAO.md)**: Detalhes da organização do projeto

### Análises Técnicas
- **[ANALISE_RESULTADOS.md](docs/ANALISE_RESULTADOS.md)**: Análise detalhada dos resultados
- **[COMPARACAO_V1_V2.md](docs/COMPARACAO_V1_V2.md)**: Comparação entre versões iniciais
- **[V3_ENHANCED_EXPLICACAO.md](docs/V3_ENHANCED_EXPLICACAO.md)**: Explicação da v3
- **[V4_EXPLICACAO.md](docs/V4_EXPLICACAO.md)**: Explicação da v4

---

## 🔍 Destaques dos Modelos

### Random Forest (Recomendado para Produção)
- ✅ **Melhor ROC-AUC**: 80.83%
- ✅ **Balanceamento**: 52.31% recall, 33.71% precision
- ✅ **Estabilidade**: CV 81.00% ± 1.17%
- 📊 **Features mais importantes**: stop_event_rate (13.17%), stop_density (12.05%)

### CatBoost (Melhor Recall)
- ✅ **Alto Recall**: 61.66% (melhor detecção de conversões)
- ✅ **ROC-AUC**: 80.40%
- ✅ **Velocidade**: Treinamento rápido (6.9s)
- 🎯 **Quando usar**: Maximizar detecção de conversões, tolerar falsos positivos

### Stacking Ensemble (Ensemble Learning)
- ✅ **Combinação**: Random Forest + CatBoost + SGD → Logistic Regression
- ✅ **ROC-AUC**: 80.48%
- ✅ **Robustez**: Combina pontos fortes de 3 modelos
- 🎯 **Quando usar**: Maximizar confiabilidade, aceitar maior complexidade

---

## 🛠️ Técnicas Aplicadas

### Tratamento de Desbalanceamento
- **Undersampling**: Proporção 2:1 (classe majoritária : classe minoritária)
- **Efeito**: Melhoria de recall de ~16% → 52% no Random Forest

### Validação Cruzada
- **Método**: TimeSeriesSplit com 5 folds
- **Métrica**: ROC-AUC (adequada para classes desbalanceadas)
- **Resultado**: Validação robusta com baixa variância (± 1-2%)

### Normalização
- **StandardScaler**: Aplicado em SVM e SGD
- **Sem normalização**: Random Forest, CatBoost, Decision Tree (robustos a escala)

---

## 📈 Próximos Passos

### Melhorias Potenciais
- [ ] Testar SMOTE (oversampling sintético) como alternativa ao undersampling
- [ ] Hyperparameter tuning com Optuna ou Grid Search
- [ ] Feature selection com SHAP values ou permutation importance
- [ ] Calibração de probabilidades (Platt scaling, isotonic regression)
- [ ] Threshold optimization para maximizar F1-Score ou métrica de negócio

### Deployment
- [ ] Criar API REST com FastAPI para servir modelos
- [ ] Containerizar com Docker
- [ ] Implementar monitoramento de drift de dados
- [ ] Configurar CI/CD para retreinamento automático
- [ ] Adicionar testes unitários para pipeline de dados

---

## 📊 Dataset

**Fonte**: BigQuery - Eventos de usuários de transporte público  
**Features**: Expanding windows (agregações temporais de eventos)  
**Target**: Conversão (0 = não converteu, 1 = converteu)  
**Desbalanceamento**: 8.87:1 (classe 0 : classe 1)

### Features Principais
- `stop_event_rate`: Taxa de eventos de parada (correlação +0.38 com target)
- `stop_density`: Densidade de paradas no período
- `stop_event_count`: Contagem total de eventos de parada
- `hour`: Hora do dia (pico de conversão às 18h com 16.81%)

---

## 🤝 Contribuindo

Para contribuir com este projeto:

1. Clone o repositório
2. Crie uma branch para sua feature (`git checkout -b feature/nova-feature`)
3. Commit suas mudanças (`git commit -m 'Adiciona nova feature'`)
4. Push para a branch (`git push origin feature/nova-feature`)
5. Abra um Pull Request

---

## 📄 Licença

Este projeto é de uso interno da Cittamobi.

---

## 👥 Autores

**Equipe de Data Science - Cittamobi**

Para dúvidas ou sugestões, consulte a documentação em `docs/` ou abra uma issue no repositório.
- **4 agregações por parada**: conversion_rate, event_count, user_frequency
- **Interações de 2ª ordem**: conversion_interaction, dist_deviation, user_stop_affinity
- **50 features** selecionadas (vs 40 no V3)

### Hiperparâmetros Vencedores:
```python
{
    'max_depth': 18,           # Árvores profundas para capturar interações complexas
    'learning_rate': 0.02,
    'min_child_weight': 3,
    'subsample': 0.85,
    'colsample_bytree': 0.85,
    'num_boost_round': 250,
    'early_stopping_rounds': 25
}
```

### Threshold Ótimo:
- **0.65** para maximizar F1-Macro (0.7760)

---

## 📈 Evolução do Projeto

### V1 - Baseline (POC)
- Primeira abordagem
- ROC-AUC: 0.8367
- Problemas: baixa precisão, muitos falsos positivos

### V2 - Enhanced
- Limpeza agressiva dos dados
- **Problema crítico**: removeu 87.9% dos dados
- ROC-AUC caiu para 0.7961

### V3 - Hybrid + Enhanced
- Limpeza moderada (11.6% removido)
- Feature selection (top 40)
- ROC-AUC: 0.9283 (+10.9% vs V1)
- Enhanced: testou 4 técnicas de balanceamento
  - Baseline (scale_pos_weight) foi o melhor

### V4 - Advanced 🏆
- Feature engineering avançado
- 5 estratégias testadas
- **Advanced Features + Deep Trees venceu**
- **ROC-AUC: 0.9731** (+16.3% vs V1)
- **Precision: 0.59** (✅ alcançou meta > 0.50)

---

## 🔬 Tecnologias Utilizadas

- **Python 3.12**
- **XGBoost**: Modelo de gradient boosting
- **Google BigQuery**: Source de dados (TABLESAMPLE 20%)
- **imbalanced-learn**: Técnicas de balanceamento (SMOTE, Tomek, etc.)
- **scikit-learn**: Métricas e validação (TimeSeriesSplit)
- **pandas, numpy**: Manipulação de dados
- **matplotlib, seaborn**: Visualizações

---

## 📚 Documentação

- **[GUIA_DE_USO.md](docs/GUIA_DE_USO.md)**: Como executar cada versão
- **[V4_EXPLICACAO.md](docs/V4_EXPLICACAO.md)**: Detalhes técnicos do V4
- **[V3_ENHANCED_EXPLICACAO.md](docs/V3_ENHANCED_EXPLICACAO.md)**: Técnicas de balanceamento
- **[ANALISE_RESULTADOS.md](docs/ANALISE_RESULTADOS.md)**: Análises detalhadas
- **[COMPARACAO_V1_V2.md](docs/COMPARACAO_V1_V2.md)**: Comparativo inicial

---

## 👨‍💻 Autor

**Stefano**  
Projeto Machine Learning - IBMEC  
Outubro 2025

---

## 📝 Notas

- Dataset: ~200k amostras (20% do total via TABLESAMPLE)
- Classe desbalanceada: ~92% classe 0, ~8% classe 1 (ratio 12:1)
- Validação temporal: TimeSeriesSplit (3 folds)
- Threshold otimizado: 0.65 (vs default 0.50)

**Modelo de produção recomendado**: V4 Advanced - Strategy 5
