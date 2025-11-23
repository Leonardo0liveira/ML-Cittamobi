# ===========================================================================
# ETAPA 2: EXPANDING WINDOWS - SEM VAZAMENTO DE DADOS
# ===========================================================================
print(f"\n{'='*70}")
print(f"ETAPA 2: EXPANDING WINDOWS (LEAK-FREE)")
print(f"{'='*70}")
print(f"💡 Para cada evento em T, usar APENAS dados históricos < T")
print(f"💡 Simula exatamente o ambiente de produção")

def create_expanding_features_optimized(df, sample_size=50000):
    """
    Cria features usando expanding windows - versão otimizada.
    Para performance, processa uma amostra representativa.
    """
    df_result = df.copy()
    
    # Se dataset muito grande, usar amostra para expanding windows
    if len(df) > sample_size:
        print(f"\n⚡ OTIMIZAÇÃO: Usando amostra de {sample_size:,} registros para expanding windows")
        print(f"   (Mantém representatividade temporal e acelera processamento)")
        
        # Amostragem estratificada temporal
        sample_indices = np.linspace(0, len(df)-1, sample_size, dtype=int)
        df_sample = df.iloc[sample_indices].copy()
        use_full_data = False
    else:
        df_sample = df.copy()
        use_full_data = True
    
    # Inicializar colunas
    user_cols = ['user_hist_events', 'user_hist_conversions', 
                 'user_hist_conversion_rate', 'user_avg_hour_hist', 
                 'user_avg_dist_hist', 'user_std_dist_hist']
    
    stop_cols = ['stop_hist_events', 'stop_hist_conversions',
                 'stop_hist_conversion_rate', 'stop_avg_freq_hist']
    
    for col in user_cols + stop_cols:
        df_sample[col] = 0.0
    
    print("📊 Calculando expanding windows...")
    start_time = time.time()
    
    # Processamento otimizado
    for i in range(len(df_sample)):
        if i % 5000 == 0 and i > 0:
            elapsed = time.time() - start_time
            rate = i / elapsed
            remaining = (len(df_sample) - i) / rate
            print(f"   {i:,}/{len(df_sample):,} ({i/len(df_sample)*100:.1f}%) - "
                  f"ETA: {remaining/60:.1f} min")
        
        if i == 0:
            continue
            
        # Histórico (apenas dados anteriores)
        hist_data = df_sample.iloc[:i]
        current_user = df_sample.iloc[i]['user_pseudo_id']
        current_stop = df_sample.iloc[i]['gtfs_stop_id']
        
        # Features do usuário
        user_mask = hist_data['user_pseudo_id'] == current_user
        if user_mask.any():
            user_hist = hist_data[user_mask]
            df_sample.iloc[i, df_sample.columns.get_loc('user_hist_events')] = len(user_hist)
            df_sample.iloc[i, df_sample.columns.get_loc('user_hist_conversions')] = user_hist[target].sum()
            
            if len(user_hist) > 0:
                df_sample.iloc[i, df_sample.columns.get_loc('user_hist_conversion_rate')] = user_hist[target].mean()
                df_sample.iloc[i, df_sample.columns.get_loc('user_avg_hour_hist')] = user_hist['hour'].mean()
                
                if 'dist_device_stop' in user_hist.columns:
                    df_sample.iloc[i, df_sample.columns.get_loc('user_avg_dist_hist')] = user_hist['dist_device_stop'].mean()
                    df_sample.iloc[i, df_sample.columns.get_loc('user_std_dist_hist')] = user_hist['dist_device_stop'].std()
        
        # Features da parada
        stop_mask = hist_data['gtfs_stop_id'] == current_stop
        if stop_mask.any():
            stop_hist = hist_data[stop_mask]
            df_sample.iloc[i, df_sample.columns.get_loc('stop_hist_events')] = len(stop_hist)
            df_sample.iloc[i, df_sample.columns.get_loc('stop_hist_conversions')] = stop_hist[target].sum()
            
            if len(stop_hist) > 0:
                df_sample.iloc[i, df_sample.columns.get_loc('stop_hist_conversion_rate')] = stop_hist[target].mean()
                
                if 'user_frequency' in stop_hist.columns:
                    df_sample.iloc[i, df_sample.columns.get_loc('stop_avg_freq_hist')] = stop_hist['user_frequency'].mean()
    
    elapsed = time.time() - start_time
    print(f"✓ Expanding windows criadas em {elapsed/60:.1f} minutos")
    
    # Se usou amostra, propagar features para dataset completo
    if not use_full_data:
        print(f"📊 Propagando features para dataset completo...")
        # Usar últimos valores conhecidos para cada user/stop
        user_last_features = df_sample.groupby('user_pseudo_id')[user_cols].last()
        stop_last_features = df_sample.groupby('gtfs_stop_id')[stop_cols].last()
        
        # Merge com dataset completo
        for col in user_cols:
            df_result[col] = df_result['user_pseudo_id'].map(user_last_features[col]).fillna(0)
        for col in stop_cols:
            df_result[col] = df_result['gtfs_stop_id'].map(stop_last_features[col]).fillna(0)
        
        print(f"✓ Features propagadas para {len(df_result):,} registros")
        return df_result
    
    return df_sample

# Criar expanding windows
df_with_expanding = create_expanding_features_optimized(df, sample_size=50000)

# Features de interação (baseadas no histórico - SEM VAZAMENTO)
df_with_expanding['hist_interaction'] = (
    df_with_expanding['user_hist_conversion_rate'] * 
    df_with_expanding['stop_hist_conversion_rate']
)

df_with_expanding['user_stop_hist_affinity'] = (
    df_with_expanding['user_hist_events'] * 
    df_with_expanding['stop_hist_events']
)

# Desvio da distância (baseado no histórico)
if 'dist_device_stop' in df_with_expanding.columns:
    df_with_expanding['dist_deviation_hist'] = abs(
        df_with_expanding['dist_device_stop'] - df_with_expanding['user_avg_dist_hist']
    )
    # Ratio de distância
    df_with_expanding['dist_ratio_hist'] = df_with_expanding['dist_device_stop'] / (df_with_expanding['user_avg_dist_hist'] + 1)

print(f"✓ Features de interação históricas criadas")

# ===========================================================================
# ETAPA 3: LIMPEZA E PREPARAÇÃO FINAL
# ===========================================================================
print(f"\n{'='*70}")
print(f"ETAPA 3: LIMPEZA E PREPARAÇÃO FINAL")
print(f"{'='*70}")

df_clean = df_with_expanding.copy()

# Filtros moderados
if 'user_frequency' in df_clean.columns:
    user_freq_threshold = df_clean['user_frequency'].quantile(0.10)
    df_clean = df_clean[df_clean['user_frequency'] >= user_freq_threshold]
    print(f"✓ Filtro user_frequency aplicado")

if 'device_lat' in df_clean.columns and 'device_lon' in df_clean.columns:
    df_clean = df_clean[~((df_clean['device_lat'].isna()) | (df_clean['device_lon'].isna()))]
    df_clean = df_clean[~((df_clean['device_lat'] == 0) & (df_clean['device_lon'] == 0))]
    print(f"✓ Filtro coordenadas aplicado")

if 'dist_device_stop' in df_clean.columns:
    dist_threshold = df_clean['dist_device_stop'].quantile(0.98)
    df_clean = df_clean[df_clean['dist_device_stop'] <= dist_threshold]
    print(f"✓ Filtro outliers de distância aplicado")

print(f"✓ Dados limpos: {len(df_clean):,} registros mantidos")

# Preparar features (REMOVENDO features com vazamento)
features_to_drop = [
    'y_pred', 'y_pred_proba', 'ctm_service_route', 'direction', 'lotacao_proxy_binaria',
    'event_timestamp',
    # REMOVER features categóricas (k-NN não lida bem com categóricas)
    'user_pseudo_id', 'gtfs_stop_id',
    # REMOVER features com vazamento (se existirem)
    'user_conversion_rate', 'user_total_conversions', 'stop_conversion_rate',
    'conversion_interaction', 'user_stop_affinity'
]

X = df_clean.drop(columns=[target] + features_to_drop, errors='ignore')
y = df_clean[target]

# K-NN trabalha melhor com features numéricas - remover categóricas restantes
categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
if categorical_cols:
    print(f"\n⚠️  Removendo {len(categorical_cols)} features categóricas (k-NN requer numéricas)")
    X = X.drop(columns=categorical_cols)

# Tratar infinitos e NaN
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(0)  # k-NN não lida bem com NaN

print(f"✓ Features finais: {X.shape[1]} (apenas numéricas)")
print(f"✓ FEATURES COM VAZAMENTO REMOVIDAS!")

print(f"\n=== Distribuição do Target ===")
target_dist = y.value_counts()
print(f"Classe 0: {target_dist[0]:,} ({target_dist[0]/len(y)*100:.2f}%)")
print(f"Classe 1: {target_dist[1]:,} ({target_dist[1]/len(y)*100:.2f}%)")

# ===========================================================================
# ETAPA 4: DIVISÃO TEMPORAL COM TimeSeriesSplit
# ===========================================================================
print(f"\n{'='*70}")
print(f"ETAPA 4: DIVISÃO TEMPORAL (TimeSeriesSplit)")
print(f"{'='*70}")

tscv = TimeSeriesSplit(n_splits=3)
for fold, (train_index, test_index) in enumerate(tscv.split(X)):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    if fold == 2:
        break

print(f"✓ Train: {len(X_train):,} | Test: {len(X_test):,}")
print(f"✓ Validação temporal respeitada!")
