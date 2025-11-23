# 📚 V4 ADVANCED - EXPLICAÇÃO DETALHADA

## 🎯 Objetivo do V4

Após análise dos resultados do V3 Enhanced, identificamos que:
- **Baseline simples venceu** todas as técnicas de balanceamento
- **Precision baixa (0.43)** é o principal problema
- **SMOTE prejudicou** o modelo (criou ruído)

O V4 foca em **técnicas avançadas** sem depender de balanceamento sintético.

---

## 🚀 5 Novas Estratégias Implementadas

### 1️⃣ **Baseline Otimizado (Referência)**
```python
scale_pos_weight = 12.05  # Razão da classe
max_depth = 12
threshold = 0.65
```

**Como funciona:**
- Usa apenas `scale_pos_weight` do XGBoost
- Árvores de profundidade moderada
- Threshold otimizado para F1-Macro

**Vantagens:**
- Simples e eficaz
- Sem overfitting
- Referência para comparação

---

### 2️⃣ **Cost-Sensitive Learning**
```python
scale_pos_weight = 12.05 * 1.5  # 50% mais peso
max_delta_step = 1  # Controla atualizações
threshold = 0.60
```

**Como funciona:**
- **Aumenta o custo** de errar na classe minoritária
- `max_delta_step=1` limita mudanças bruscas (previne overfitting)
- Threshold mais baixo (0.60) para aumentar recall

**Quando usar:**
- Quando **falsos negativos** são muito caros
- Quando precision pode ser sacrificada por recall

**Trade-offs:**
- ✅ **Recall aumenta** (detecta mais positivos)
- ❌ **Precision pode cair** (mais falsos positivos)

---

### 3️⃣ **User Frequency Undersampling** (Sugestão do Professor) ⭐
```python
# Filtrar apenas usuários frequentes (top 60%)
user_freq_threshold = quantile(0.40)

# Undersampling inteligente
minority = all_positive_samples
majority = top_frequent_users (ratio 5:1)
```

**Como funciona:**
1. **Filtrar usuários frequentes** (≥ percentil 40)
   - Remove usuários casuais/esporádicos
   - Mantém usuários engajados

2. **Undersampling da classe majoritária**
   - Mantém TODOS os positivos
   - Seleciona apenas os negativos mais relevantes
   - Prioriza usuários com maior `user_frequency`

3. **Ratio 5:1** (menos agressivo que 3:1)
   - 5 negativos para cada 1 positivo
   - Mantém mais dados que ratio 3:1

**Por que funciona:**
- **Qualidade > Quantidade**: Usuários frequentes têm padrões mais consistentes
- **Remove ruído**: Usuários casuais podem ter comportamento aleatório
- **Preserva informação**: Mantém 100% dos positivos

**Vantagens:**
- ✅ Reduz ruído do dataset
- ✅ Mantém amostras de alta qualidade
- ✅ Treino mais rápido (menos dados)
- ✅ Generaliza melhor

**Desvantagens:**
- ❌ Perde informação de usuários casuais
- ❌ Pode não funcionar se usuários casuais também convertem

---

### 4️⃣ **Ensemble Stacking**
```python
# 3 modelos com configurações diferentes
Model 1: Conservador  (precision ↑, max_depth=8)
Model 2: Agressivo    (recall ↑, max_depth=15, weight*2)
Model 3: Balanceado   (F1 ↑, max_depth=12, weight*1.3)

# Votação ponderada
final_prediction = mean([prob1, prob2, prob3])
```

**Como funciona:**
1. **Treina 3 modelos diferentes:**
   - **Conservador**: Alta precision, poucas predições positivas
   - **Agressivo**: Alta recall, muitas predições positivas
   - **Balanceado**: Meio-termo

2. **Combina probabilidades:**
   - Média aritmética das 3 probabilidades
   - Suaviza predições extremas

**Por que funciona:**
- **Diversidade**: Cada modelo captura padrões diferentes
- **Reduz variance**: Erros individuais se cancelam
- **Robustez**: Menos sensível a outliers

**Vantagens:**
- ✅ Geralmente melhor que modelos individuais
- ✅ Mais robusto
- ✅ Captura diferentes aspectos dos dados

**Desvantagens:**
- ❌ 3x mais lento para treinar
- ❌ 3x mais memória
- ❌ Mais complexo para deployment

---

### 5️⃣ **Advanced Features + Deep Trees**
```python
max_depth = 18  # Árvores mais profundas
min_child_weight = 3  # Menos restrição
num_boost_round = 250  # Mais iterações
```

**Features Avançadas Criadas:**

#### **Agregações por Usuário:**
```python
user_conversion_rate    # Taxa de conversão histórica
user_total_conversions  # Total de conversões
user_total_events       # Frequência total
user_avg_dist          # Distância média
user_std_dist          # Variabilidade de distância
user_min/max_dist      # Range de distância
user_avg_hour          # Hora média de uso
user_std_hour          # Variabilidade temporal
```

**Por que ajudam:**
- Capturam **padrões históricos** do usuário
- Usuário que converte 80% das vezes → alta probabilidade
- Usuário com distância consistente → comportamento previsível

#### **Agregações por Parada:**
```python
stop_conversion_rate      # Taxa de conversão na parada
stop_event_count_agg     # Popularidade da parada
stop_user_freq_mean      # Frequência média dos usuários
stop_user_freq_median    # Frequência mediana
```

**Por que ajudam:**
- Paradas com alta conversão → mais propensas
- Paradas populares → padrões mais estáveis

#### **Interações de 2ª Ordem:**
```python
# Interação usuário x parada
conversion_interaction = user_rate * stop_rate

# Desvio de comportamento
dist_deviation = |atual - média_usuário|
dist_ratio = atual / média_usuário

# Afinidade usuário-parada
user_stop_affinity = user_freq * stop_events
```

**Por que ajudam:**
- **Captura sinergias**: Usuário bom + Parada boa = excelente
- **Detecta anomalias**: Distância muito diferente da média → suspeito
- **Afinidade**: Usuário frequente em parada popular → alta conversão

#### **Árvores Profundas:**
```python
max_depth = 18  # vs 12 no baseline
```

**Vantagens:**
- Captura interações complexas entre features
- Aprende padrões não-lineares profundos

**Desvantagens:**
- ⚠️ **Risco de overfitting** (cuidado!)
- Por isso usamos `early_stopping_rounds=25`

---

## 📊 Comparação: Qual Estratégia Escolher?

| Estratégia | Quando Usar | Vantagem Principal |
|-----------|-------------|-------------------|
| **Baseline** | Sempre comece aqui | Simples, eficaz, rápido |
| **Cost-Sensitive** | Falsos negativos muito caros | Aumenta recall |
| **User Freq Undersampling** | Dataset ruidoso, usuários casuais | Remove ruído, alta qualidade |
| **Ensemble** | Produção, precisão crítica | Mais robusto |
| **Deep Trees + Features** | Muitos dados, relações complexas | Captura padrões complexos |

---

## 🎓 Conceitos Importantes

### **Precision vs Recall Trade-off**
```
Precision = VP / (VP + FP)  # Das predições positivas, quantas corretas?
Recall = VP / (VP + FN)     # Dos reais positivos, quantos capturei?

↑ Precision → Menos FP → Menos falsos alarmes
↑ Recall → Menos FN → Não perco positivos verdadeiros
```

**V3 tinha:**
- Precision = 0.43 (de 100 predições positivas, só 43 eram corretas)
- Recall = 0.47 (de 100 positivos reais, detectamos 47)

### **F1-Macro vs F1-Score**
```
F1-Score = média harmônica entre Precision e Recall (classe 1)
F1-Macro = (F1_classe_0 + F1_classe_1) / 2

F1-Macro é melhor para classes desbalanceadas!
```

**Por quê?**
- F1-Score ignora performance na classe majoritária
- F1-Macro força o modelo a ser bom em AMBAS as classes

---

## 🔬 Experimento: O que Testar Agora?

### **Próximos Passos:**

1. **Execute o V4:**
```bash
conda activate cittamobi-forecast
python model_v4_advanced.py
```

2. **Compare os resultados:**
- Qual estratégia teve maior F1-Macro?
- User Frequency Undersampling funcionou?
- Ensemble melhorou?

3. **Analise trade-offs:**
- Se Precision aumentou → Ótimo! Menos falsos alarmes
- Se Recall caiu muito → Talvez não vale a pena

---

## 💡 Insights Esperados

### **User Frequency Undersampling deve:**
- ✅ **Aumentar Precision** (dados mais limpos)
- ✅ **Manter ou aumentar ROC-AUC**
- ❓ **Recall pode variar** (depende da qualidade dos usuários frequentes)

### **Ensemble deve:**
- ✅ **Estabilizar métricas** (menos variance)
- ✅ **Pequeno ganho em todas as métricas**
- ✅ **ROC-AUC ligeiramente melhor**

### **Deep Trees + Features deve:**
- ✅ **Melhor performance se houver padrões complexos**
- ⚠️ **Risco de overfitting** (validar no test set!)

---

## 📈 Critério de Sucesso

**V4 é melhor que V3 se:**
1. **F1-Macro > 0.7143** (baseline do V3)
2. **Precision > 0.43** (problema principal)
3. **ROC-AUC ≥ 0.9324** (manter qualidade geral)

**Melhoria ideal:**
- Precision: 0.43 → **0.50+** (16% improvement)
- F1-Macro: 0.7143 → **0.73+** (2% improvement)
- Manter Recall ≥ 0.45

---

## 🎯 Conclusão

O V4 explora técnicas mais sofisticadas que vão além de balanceamento simples:

1. **Custo assimétrico** - penaliza erros na minoria
2. **Filtragem inteligente** - qualidade > quantidade
3. **Ensemble** - combina múltiplas visões
4. **Feature engineering** - captura padrões complexos

**Próximo passo:** Execute e veja qual estratégia vence! 🚀
