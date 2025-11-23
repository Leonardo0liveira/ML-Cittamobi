"""
Teste rápido do ambiente V7
"""
import sys
print("Python version:", sys.version)

try:
    import pandas as pd
    print("✓ pandas importado")
except ImportError as e:
    print("✗ pandas:", e)

try:
    import numpy as np
    print("✓ numpy importado")
except ImportError as e:
    print("✗ numpy:", e)

try:
    from google.cloud import bigquery
    print("✓ bigquery importado")
except ImportError as e:
    print("✗ bigquery:", e)

try:
    import lightgbm as lgb
    print("✓ lightgbm importado")
except ImportError as e:
    print("✗ lightgbm:", e)

try:
    import xgboost as xgb
    print("✓ xgboost importado")
except ImportError as e:
    print("✗ xgboost:", e)

try:
    import sklearn
    print("✓ sklearn importado")
except ImportError as e:
    print("✗ sklearn:", e)

try:
    import holidays
    print("✓ holidays importado")
except ImportError as e:
    print("✗ holidays:", e)

print("\nTodas as bibliotecas necessárias estão disponíveis!")
