"""
═══════════════════════════════════════════════════════════════════════════════
SCRIPT DE INFERÊNCIA - MODEL V8 PRODUCTION
═══════════════════════════════════════════════════════════════════════════════

Este script demonstra como carregar e usar o modelo em produção.

Uso:
    python inference_v8_production.py

═══════════════════════════════════════════════════════════════════════════════
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
import pickle
import json
from datetime import datetime

class CittamobiConversionPredictor:
    """
    Classe para realizar predições de conversão usando o Model V8 Production.
    """
    
    def __init__(self, model_path='./'):
        """
        Inicializa o preditor carregando todos os artefatos necessários.
        
        Args:
            model_path (str): Caminho para os arquivos do modelo
        """
        print("🔧 Inicializando Cittamobi Conversion Predictor...")
        
        # Carregar LightGBM
        self.lgb_model = lgb.Booster(
            model_file=f'{model_path}/lightgbm_model_v8_production.txt'
        )
        print("   ✓ LightGBM carregado")
        
        # Carregar XGBoost
        self.xgb_model = xgb.Booster()
        self.xgb_model.load_model(f'{model_path}/xgboost_model_v8_production.json')
        print("   ✓ XGBoost carregado")
        
        # Carregar Scaler
        with open(f'{model_path}/scaler_v8_production.pkl', 'rb') as f:
            self.scaler = pickle.load(f)
        print("   ✓ Scaler carregado")
        
        # Carregar Features
        with open(f'{model_path}/selected_features_v8_production.txt', 'r') as f:
            self.feature_cols = [line.strip() for line in f]
        print(f"   ✓ {len(self.feature_cols)} features carregadas")
        
        # Carregar Configuração
        with open(f'{model_path}/model_config_v8_production.json', 'r') as f:
            self.config = json.load(f)
        print("   ✓ Configuração carregada")
        
        # Pesos do ensemble
        self.w_lgb = self.config['ensemble_weights']['lightgbm']
        self.w_xgb = self.config['ensemble_weights']['xgboost']
        
        print(f"\n✅ Preditor inicializado com sucesso!")
        print(f"   Versão: {self.config['model_version']}")
        print(f"   Data: {self.config['creation_date']}")
        print(f"   F1 Classe 1: {self.config['metrics']['f1_class_1']:.4f}")
        print(f"   ROC-AUC: {self.config['metrics']['roc_auc']:.4f}")
        print()
    
    def get_dynamic_threshold(self, stop_conversion_rate):
        """
        Calcula o threshold dinâmico baseado na taxa de conversão da parada.
        
        Args:
            stop_conversion_rate (float): Taxa de conversão histórica (0-1)
            
        Returns:
            float: Threshold a ser usado (0.40, 0.50, 0.60 ou 0.75)
        """
        rules = self.config['threshold_rules']
        
        if stop_conversion_rate >= rules['high_conversion']['min']:
            return rules['high_conversion']['threshold']
        elif stop_conversion_rate >= rules['medium_conversion']['min']:
            return rules['medium_conversion']['threshold']
        elif stop_conversion_rate >= rules['low_conversion']['min']:
            return rules['low_conversion']['threshold']
        else:
            return rules['very_low_conversion']['threshold']
    
    def predict(self, df, return_proba=False):
        """
        Realiza predições no DataFrame fornecido.
        
        Args:
            df (pd.DataFrame): DataFrame com as features necessárias
            return_proba (bool): Se True, retorna também as probabilidades
            
        Returns:
            np.array ou tuple: Predições (e probabilidades se return_proba=True)
        """
        # Validar features
        missing_features = set(self.feature_cols) - set(df.columns)
        if missing_features:
            raise ValueError(f"Features faltando: {missing_features}")
        
        # Selecionar e ordenar features
        X = df[self.feature_cols].copy()
        
        # Normalizar
        X_scaled = self.scaler.transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=self.feature_cols)
        
        # Predições LightGBM
        pred_lgb = self.lgb_model.predict(X_scaled)
        
        # Predições XGBoost
        pred_xgb = self.xgb_model.predict(xgb.DMatrix(X_scaled))
        
        # Ensemble
        pred_ensemble = self.w_lgb * pred_lgb + self.w_xgb * pred_xgb
        
        # Aplicar threshold dinâmico
        if 'stop_historical_conversion' in df.columns:
            thresholds = df['stop_historical_conversion'].apply(
                self.get_dynamic_threshold
            ).values
        else:
            # Fallback: usar threshold padrão
            thresholds = np.full(len(df), 0.60)
        
        # Predições finais
        predictions = (pred_ensemble > thresholds).astype(int)
        
        if return_proba:
            return predictions, pred_ensemble, thresholds
        else:
            return predictions
    
    def predict_single(self, event_data):
        """
        Realiza predição para um único evento.
        
        Args:
            event_data (dict): Dicionário com as features do evento
            
        Returns:
            dict: Resultado da predição com probabilidade e threshold
        """
        # Converter para DataFrame
        df = pd.DataFrame([event_data])
        
        # Fazer predição
        prediction, proba, threshold = self.predict(df, return_proba=True)
        
        return {
            'predicted_conversion': int(prediction[0]),
            'conversion_probability': float(proba[0]),
            'threshold_used': float(threshold[0]),
            'confidence': abs(proba[0] - threshold[0])
        }


def main():
    """
    Exemplo de uso do preditor.
    """
    print("="*80)
    print("🚀 EXEMPLO DE USO - CITTAMOBI CONVERSION PREDICTOR")
    print("="*80)
    print()
    
    # Inicializar preditor
    predictor = CittamobiConversionPredictor(model_path='.')
    
    print("="*80)
    print("📝 EXEMPLO 1: Predição Individual")
    print("="*80)
    print()
    
    # Exemplo de evento (valores fictícios para demonstração)
    event_example = {
        'stop_historical_conversion': 0.35,
        'stop_density': 45.2,
        'dist_to_nearest_cbd': 12.5,
        'stop_cluster': 2,
        'cluster_conversion_rate': 0.32,
        'stop_volatility': 0.15,
        'hour_conversion_rate': 0.25,
        'dow_conversion_rate': 0.28,
        'stop_hour_conversion': 0.30,
        'geo_temporal': 12.5,
        'density_peak': 45.2,
        'user_conversion_rate': 0.40,
        'user_vs_stop_ratio': 0.6,
        'stop_rarity': 0.001,
        'user_rarity': 0.005,
        'stop_dist_std': 0.02,
        # ... outras 29 features base
    }
    
    # Para demonstração, vamos criar um DataFrame completo com valores dummy
    # Em produção, você teria todas as 45 features
    dummy_data = {col: 0.0 for col in predictor.feature_cols}
    dummy_data.update(event_example)
    
    # Predição individual
    result = predictor.predict_single(dummy_data)
    
    print(f"Evento de exemplo:")
    print(f"   Taxa de conversão da parada: {event_example['stop_historical_conversion']:.1%}")
    print(f"   Distância ao CBD: {event_example['dist_to_nearest_cbd']:.1f}km")
    print()
    print(f"Resultado da predição:")
    print(f"   ✓ Conversão prevista: {'SIM' if result['predicted_conversion'] else 'NÃO'}")
    print(f"   ✓ Probabilidade: {result['conversion_probability']:.2%}")
    print(f"   ✓ Threshold usado: {result['threshold_used']:.2f}")
    print(f"   ✓ Confiança: {result['confidence']:.2%}")
    print()
    
    print("="*80)
    print("📝 EXEMPLO 2: Predição em Batch (simulado)")
    print("="*80)
    print()
    
    # Criar DataFrame com múltiplos eventos (simulado)
    n_samples = 100
    batch_data = pd.DataFrame({col: np.random.randn(n_samples) for col in predictor.feature_cols})
    batch_data['stop_historical_conversion'] = np.random.uniform(0, 0.8, n_samples)
    
    # Predições
    predictions, probas, thresholds = predictor.predict(batch_data, return_proba=True)
    
    print(f"Batch de {n_samples} eventos processados:")
    print(f"   ✓ Conversões previstas: {predictions.sum()} ({predictions.mean():.1%})")
    print(f"   ✓ Probabilidade média: {probas.mean():.2%}")
    print(f"   ✓ Distribuição de thresholds:")
    for t in [0.40, 0.50, 0.60, 0.75]:
        count = (thresholds == t).sum()
        pct = count / len(thresholds) * 100
        print(f"      - {t:.2f}: {count} eventos ({pct:.1f}%)")
    print()
    
    print("="*80)
    print("✅ EXEMPLOS CONCLUÍDOS COM SUCESSO!")
    print("="*80)
    print()
    print("📚 Para uso em produção:")
    print("   1. Carregar seus dados reais")
    print("   2. Garantir que todas as 45 features estão presentes")
    print("   3. Executar predictor.predict(df)")
    print("   4. Usar as predições para tomada de decisão")
    print()


if __name__ == "__main__":
    main()
