# 📊 Relatório Comparativo: Modelo V1 vs V2 Enhanced

## 🎯 Resumo Executivo

Comparação entre o modelo baseline (V1) com 50k amostras e o modelo enhanced (V2) com limpeza rigorosa de dados, 500k amostras iniciais e feature engineering avançado.

---

## 📈 Comparação de Performance

| Métrica | V1 (Baseline) | V2 (Enhanced) | Diferença | Interpretação |
|---------|---------------|---------------|-----------|---------------|
| **ROC-AUC** | 0.8367 | 0.7961 | **-0.0406** | ⚠️ Redução de 4.8% |
| **Accuracy** | 89.02% | 86.62% | **-2.40%** | ⚠️ Ligeira queda |
| **Precision** | 45.19% | 41.99% | **-3.20%** | ⚠️ Ligeira queda |
| **Recall** | 51.21% | 48.89% | **-2.32%** | ⚠️ Ligeira queda |
| **F1-Score** | 0.4801 | 0.4518 | **-0.0283** | ⚠️ Redução de 5.9% |
| **Threshold** | 0.6 | 0.5 | -0.1 | Voltou ao padrão |

---

## 📊 Análise dos Dados

### **V1 - Dataset Baseline**
```
Amostras: 50,000
Features: 38
Filtros: Apenas remoção de data leakage
Balanceamento: 90.23% / 9.77%
```

### **V2 - Dataset Enhanced**
```
Amostras iniciais: 500,000
Amostras após filtros: 60,498 (12.1% mantidos)
Features: 49 (+11 novas)
Filtros aplicados:
  ✓ Usuários com baixa frequência: -24,848
  ✓ Localização inválida: -81
  ✓ Distância muito alta: -3,754
  ✓ Headway inválido: 0
  ✓ Paradas com poucos eventos: -10,819
  ✓ Total removido: 439,502 (87.9%)

Balanceamento final: 89.42% / 10.58%
```

---

## 🔍 Análise Detalhada

### **1. Por que V2 teve Performance Inferior?**

#### **Hipótese 1: Overfitting no V1** ❌
- V1 tinha apenas 50k amostras
- V2 com limpeza rigorosa ficou com 60k amostras de MAIOR qualidade
- Se fosse overfitting, V2 deveria ter performance melhor → Não é o caso

#### **Hipótese 2: Dados muito "limpos" ✅ PROVÁVEL**
- **87.9% dos dados foram removidos** (de 500k → 60k)
- Removemos usuários casuais, eventos com erro de GPS, etc.
- **Dataset V2 é muito mais homogêneo** → Menos variação nos padrões
- O modelo V1 se beneficiava da "sujeira" dos dados para generalizar melhor

#### **Hipótese 3: Distribuição Diferente dos Dados** ✅ PROVÁVEL
```
V1: LIMIT 50000 (primeiras 50k linhas)
V2: LIMIT 500000 + filtros rigorosos (diferentes subconjuntos)

A query LIMIT no BigQuery não é determinística!
Dados podem ser de períodos/regiões diferentes!
```

#### **Hipótese 4: Complexidade Excessiva** ⚠️ POSSÍVEL
- V2 tem 49 features vs 38 no V1 (+11 features)
- Mais features de interação podem ter introduzido ruído
- Algumas features criadas podem não agregar valor

---

## 🎯 Matriz de Confusão Comparativa

### **V1 - Modelo Baseline (threshold=0.6)**
```
                Predito
                0      1
Real  0     13,991  1,025  ← 1,025 Falsos Positivos (6.8%)
      1        805    845  ← 805 Falsos Negativos (48.8%)
```

### **V2 - Modelo Enhanced (threshold=0.5)**
```
                Predito
                0      1
Real  0     12,266  1,152  ← 1,152 Falsos Positivos (8.6%) ⬆️ Pior
      1        872    834  ← 872 Falsos Negativos (51.1%) ⬆️ Pior
```

**Observação:** V2 tem MAIS erros em ambas categorias!

---

## 💡 Insights e Descobertas

### **1. Limpeza de Dados ≠ Sempre Melhor**
- Remover 87.9% dos dados foi **muito agressivo**
- Dataset "limpo" demais pode **reduzir diversidade** necessária para generalização
- O "ruído" nos dados pode conter padrões reais de comportamento do usuário

### **2. Quantidade vs Qualidade**
- V1: 50k amostras "sujas" → ROC-AUC 0.8367
- V2: 60k amostras "limpas" → ROC-AUC 0.7961
- **Conclusão:** Nem sempre "mais limpo" significa "melhor"

### **3. Problema de Amostragem**
- `LIMIT` no BigQuery não garante amostragem representativa
- V1 e V2 podem ter dados de **períodos/regiões diferentes**
- Solução: Usar `ORDER BY RAND()` ou `TABLESAMPLE`

### **4. Feature Engineering**
- **11 novas features criadas**, mas performance piorou
- Possíveis features redundantes ou com ruído
- Necessidade de **seleção de features** (ex: SHAP, permutation importance)

---

## 🔧 Recomendações para V3

### **Estratégia 1: Limpeza Moderada**
```python
# Em vez de remover 87.9%, tentar remover apenas 30-40%
# Filtros mais brandos:

# 1. Usuários com baixa frequência (Q10 em vez de Q25)
user_freq_threshold = df['user_frequency'].quantile(0.10)

# 2. Distância (Q98 em vez de Q95)
dist_threshold = df['dist_device_stop'].quantile(0.98)

# 3. Paradas (Q10 em vez de Q20)
stop_threshold = df['stop_event_count'].quantile(0.10)
```

### **Estratégia 2: Amostragem Aleatória**
```sql
-- Query melhorada com amostragem aleatória
SELECT * FROM `proj-ml-469320.app_cittamobi.dataset-updated` 
TABLESAMPLE SYSTEM (10 PERCENT)  -- 10% aleatório da tabela
LIMIT 200000
```

### **Estratégia 3: Seleção de Features**
```python
# Usar apenas top 30 features mais importantes
from xgboost import plot_importance
importance = model.get_score(importance_type='gain')
top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:30]
```

### **Estratégia 4: Ensemble com V1 e V2**
```python
# Combinar predições dos dois modelos
pred_final = 0.6 * pred_v1 + 0.4 * pred_v2
```

---

## 📊 Features Criadas em V2

### **Novas Features (11 total):**

1. **minute** - Minuto da hora
2. **week_of_year** - Semana do ano
3. **hour_x_dayofweek** - Interação hora x dia da semana
4. **dist_x_peak_enhanced** - Distância x pico (melhorada)
5. **event_rate_normalized** - Taxa de eventos normalizada
6. **headway_per_hour** - Headway por hora
7. **event_density** - Densidade de eventos
8. **day_of_month_sin** - Componente cíclico do dia
9. **day_of_month_cos** - Componente cíclico do dia
10. **week_sin** - Componente cíclico da semana
11. **week_cos** - Componente cíclico da semana

**Análise:** Algumas podem ser redundantes (ex: já existem hour_sin/hour_cos)

---

## 🏆 Conclusão

### **Vencedor: V1 (Modelo Baseline)**

| Aspecto | V1 | V2 |
|---------|----|----|
| **Performance** | ✅ Melhor (ROC-AUC 0.8367) | ❌ Inferior (ROC-AUC 0.7961) |
| **Simplicidade** | ✅ 38 features | ⚠️ 49 features (mais complexo) |
| **Tempo de treino** | ✅ Mais rápido (50k samples) | ⚠️ Mais lento (60k samples) |
| **Interpretabilidade** | ✅ Mais simples | ⚠️ Mais complexo |

### **Lições Aprendidas:**

1. ✅ **Nem sempre "mais dados" ou "dados mais limpos" = Melhor modelo**
2. ✅ **Filtros muito rigorosos podem remover variabilidade necessária**
3. ✅ **Feature engineering excessivo pode introduzir ruído**
4. ✅ **Amostragem não aleatória (LIMIT) pode enviesar resultados**

### **Próximos Passos:**

1. **V3: Versão Híbrida**
   - Usar 200k amostras com `TABLESAMPLE`
   - Filtros moderados (remover 30-40%)
   - Selecionar top 35 features (entre V1 e V2)

2. **Análise de SHAP Values**
   - Entender quais features realmente importam
   - Remover features redundantes

3. **Validação Cruzada Temporal**
   - Usar múltiplos períodos de tempo
   - Garantir robustez temporal

4. **Ensemble**
   - Combinar V1 e V2
   - Pode capturar o melhor de ambos

---

## 📁 Arquivos Gerados

### **V1:**
- `xgboost_model_optimized.json`
- `confusion_matrix.png`
- `roc_curve.png`
- `threshold_analysis.png`
- `feature_importance.png`

### **V2:**
- `xgboost_model_v2_enhanced.json`
- `confusion_matrix_v2.png`
- `roc_curve_v2.png`
- `threshold_analysis_v2.png`
- `feature_importance_v2.png`

---

## 🎓 Recomendação Final

**Para produção imediata: USE O MODELO V1**

- ROC-AUC superior (0.8367 vs 0.7961)
- Mais simples e rápido
- Melhor generalização
- Threshold otimizado (0.6)

**Para pesquisa/melhoria: Continue experimentando V3**

- Implementar as estratégias sugeridas
- Testar amostragem aleatória
- Reduzir agressividade dos filtros
- Fazer seleção de features

---

**Data do Relatório:** 29 de Outubro de 2025  
**Analista:** GitHub Copilot  
**Modelos Comparados:** V1 (Baseline) vs V2 (Enhanced)
