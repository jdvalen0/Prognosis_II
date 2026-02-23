"""
Test de integración end-to-end
Valida que el flujo completo funcione desde ingesta hasta predicción.
"""

import unittest
import pandas as pd
import numpy as np
import os
import tempfile
from src.config import SystemConfig
from src.data.db_manager import DatabaseManager
from src.data.preprocessor import DataPreprocessor
from src.features.selector import KeyVariableSelector
from src.models.baseline_modeler import BaselineModeler
from src.models.predictor import IndustrialFailurePredictor


class TestEndToEndPipeline(unittest.TestCase):
    """Test de integración completo."""
    
    @classmethod
    def setUpClass(cls):
        """Configuración inicial para todos los tests."""
        cls.config = SystemConfig()
        # Usar base de datos de prueba si está disponible
        cls.config.DB_NAME = "prognosis_test_db"
        
    def setUp(self):
        """Configuración para cada test."""
        # Crear datos sintéticos para pruebas
        np.random.seed(42)
        n_samples = 200
        
        # Generar serie temporal con seasonality
        t = np.arange(n_samples)
        seasonal = 10 + 5 * np.sin(2 * np.pi * t / 24)  # Seasonality de 24 horas
        noise = np.random.randn(n_samples) * 0.5
        trend = 0.01 * t
        
        self.test_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=n_samples, freq='H'),
            'voltage_l1': seasonal + noise + trend,
            'current_l1': 5 + 2 * np.sin(2 * np.pi * t / 24) + np.random.randn(n_samples) * 0.3,
            'temperature': 25 + 3 * np.cos(2 * np.pi * t / 12) + np.random.randn(n_samples) * 0.2,
            'constant_var': [10.0] * n_samples  # Variable constante (debe eliminarse)
        })
        
        # Crear archivo temporal
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        self.test_data.to_csv(self.temp_file.name, index=False)
        self.temp_file.close()
        
    def tearDown(self):
        """Limpieza después de cada test."""
        if os.path.exists(self.temp_file.name):
            os.unlink(self.temp_file.name)
            
    @unittest.skipIf(os.getenv('SKIP_DB_TESTS') == '1', "Saltando tests que requieren DB")
    def test_full_pipeline_without_db(self):
        """Test: Pipeline completo sin requerir DB real (mocks)."""
        from unittest.mock import MagicMock
        
        # Mock de DB
        db_mock = MagicMock()
        db_mock.engine = MagicMock()
        db_mock.get_baseline_stats = MagicMock(return_value={})
        db_mock.save_selection_metrics = MagicMock()
        db_mock.save_model_version = MagicMock()
        
        # Preprocesador
        preprocessor = DataPreprocessor(db_mock)
        clean_data = preprocessor.clean_data(self.test_data)
        normalized_data = preprocessor.normalize_data(clean_data)
        
        # Verificar que variable constante fue eliminada
        self.assertNotIn('constant_var', clean_data.columns)
        self.assertGreater(len(normalized_data.columns), 0)
        
        # Selector
        selector = KeyVariableSelector(self.config, db_mock)
        critical_vars = selector.select_critical_variables(clean_data)
        
        # Debe identificar al menos una variable crítica
        self.assertGreater(len(critical_vars), 0)
        
        # Baseline Modeler
        modeler = BaselineModeler(self.config, db_mock)
        
        # Entrenar para una variable crítica
        if critical_vars:
            var = critical_vars[0]
            if var in normalized_data.columns:
                results = modeler.fit_ensemble(var, normalized_data[var])
                
                # Verificar que se entrenaron modelos
                self.assertIn('models', results)
                self.assertIn('limits', results)
                
                # Verificar límites robustos (percentiles)
                limits = results['limits']
                self.assertIn('baseline', limits)
                self.assertIn('upper', limits)
                self.assertIn('lower', limits)
                
                # Guardar baseline
                modeler.save_baseline(results)
                
                # Predictor
                predictor = IndustrialFailurePredictor(self.config, modeler)
                
                # Predicción sobre datos recientes
                recent_data = normalized_data.tail(24)  # Últimas 24 horas
                prediction = predictor.predict(recent_data)
                
                # Verificar estructura de predicción
                self.assertIn('system_health', prediction)
                self.assertIn('variable_risks', prediction)
                self.assertIn('top_influencers', prediction)
                self.assertIn('alerts', prediction)
                
                # Verificar que top_influencers tiene formato correcto (corrección aplicada)
                for item in prediction['top_influencers']:
                    self.assertEqual(len(item), 3)  # (var, risk, explanation)
                    
    def test_pipeline_robustness_edge_cases(self):
        """Test: Pipeline maneja casos edge correctamente."""
        from unittest.mock import MagicMock
        
        db_mock = MagicMock()
        db_mock.engine = MagicMock()
        db_mock.get_baseline_stats = MagicMock(return_value={})
        db_mock.save_selection_metrics = MagicMock()
        db_mock.save_model_version = MagicMock()
        
        # Caso 1: Datos muy pequeños
        small_data = self.test_data.head(10)
        preprocessor = DataPreprocessor(db_mock)
        clean_data = preprocessor.clean_data(small_data)
        
        # No debe fallar
        self.assertIsNotNone(clean_data)
        
        # Caso 2: Datos con muchos NaN
        data_with_nan = self.test_data.copy()
        data_with_nan.loc[0:50, 'voltage_l1'] = np.nan
        
        clean_data = preprocessor.clean_data(data_with_nan)
        normalized_data = preprocessor.normalize_data(clean_data)
        
        # Debe manejar NaN correctamente
        self.assertFalse(normalized_data['voltage_l1'].isna().any())
        
    def test_xai_integration_in_pipeline(self):
        """Test: XAI se integra correctamente en el pipeline."""
        from unittest.mock import MagicMock
        
        db_mock = MagicMock()
        db_mock.engine = MagicMock()
        db_mock.get_baseline_stats = MagicMock(return_value={})
        db_mock.save_selection_metrics = MagicMock()
        db_mock.save_model_version = MagicMock()
        
        preprocessor = DataPreprocessor(db_mock)
        clean_data = preprocessor.clean_data(self.test_data)
        normalized_data = preprocessor.normalize_data(clean_data)
        
        selector = KeyVariableSelector(self.config, db_mock)
        critical_vars = selector.select_critical_variables(clean_data)
        
        modeler = BaselineModeler(self.config, db_mock)
        
        if critical_vars:
            var = critical_vars[0]
            if var in normalized_data.columns:
                results = modeler.fit_ensemble(var, normalized_data[var])
                modeler.save_baseline(results)
                
                predictor = IndustrialFailurePredictor(self.config, modeler)
                
                # Simular múltiples ciclos para entrenar XAI
                for i in range(60):  # 60 ciclos para tener suficientes datos
                    recent_data = normalized_data.iloc[[i % len(normalized_data)]]
                    prediction = predictor.predict(recent_data)
                    
                # Verificar que XAI está funcionando
                self.assertIn('shap_explanations', prediction)
                
                # Después de 60 ciclos, debería haber explicaciones SHAP
                if len(predictor.xai_explainer.training_data) >= 50:
                    self.assertGreater(len(prediction['shap_explanations']), 0)


if __name__ == '__main__':
    unittest.main()
