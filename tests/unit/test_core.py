import unittest
import pandas as pd
import numpy as np
import os
import logging
from src.config import SystemConfig
from src.data.db_manager import DatabaseManager
from src.data.preprocessor import DataPreprocessor
from src.features.selector import KeyVariableSelector

class TestPrognosisPipeline(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.config = SystemConfig()
        cls.db = DatabaseManager(cls.config)
        # No necesitamos la DB real para las pruebas unitarias simples
        
    def test_preprocessor_robust_scaling(self):
        preprocessor = DataPreprocessor(self.db)
        # Datos con outliers extremos
        data = pd.DataFrame({
            'signal': [10, 11, 12, 11, 10, 1000],  # 1000 es outlier
            'timestamp': pd.date_range("2025-01-01", periods=6, freq='h')
        })
        
        cleaned = preprocessor.clean_data(data)
        normalized = preprocessor.normalize_data(cleaned)
        
        # Con RobustScaler, el outlier 1000 no debería afectar la escala de los otros 
        # tanto como lo haría StandardScaler (Z-score)
        self.assertTrue(normalized['signal'].iloc[0] < 1.0)

    def test_selector_dynamism_logic(self):
        class DummyDB:
            def save_selection_metrics(self, metrics):
                return None

        selector = KeyVariableSelector(self.config, DummyDB())

        # Variable dinámica vs casi-constante con ruido mínimo
        np.random.seed(0)
        data = pd.DataFrame({
            'dynamic': np.sin(np.linspace(0, 10, 100)),
            'steady': 10.0 + (np.random.randn(100) * 1e-4)
        })
        
        selector.select_critical_variables(data)
        scores = selector.results.get('variables', {})
        self.assertIn('dynamic', scores)
        self.assertIn('steady', scores)

        score_dynamic = scores['dynamic']['final_score']
        score_steady = scores['steady']['final_score']
        
        self.assertTrue(score_dynamic > score_steady)
        logging.info("Selector validado: dynamic puntúa más que steady.")

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    unittest.main()
