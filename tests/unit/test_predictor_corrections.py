"""
Tests para validar las correcciones aplicadas en IndustrialFailurePredictor
Específicamente: zip seguro, validación de NaN, TTF estimation
"""

import unittest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from src.models.predictor import IndustrialFailurePredictor
from src.config import SystemConfig


class TestPredictorCorrections(unittest.TestCase):
    """Tests para validar correcciones críticas en el predictor."""
    
    def setUp(self):
        """Configuración inicial."""
        self.config = SystemConfig()
        
        # Mock del baseline_learner
        self.baseline_mock = MagicMock()
        self.baseline_mock.results = {
            'adaptive_limits': {
                'var1': {
                    'baseline': 10.0,
                    'upper': 15.0,
                    'lower': 5.0,
                    'p99': 16.0,
                    'p01': 4.0
                }
            }
        }
        self.baseline_mock.get_forecast.return_value = 10.5
        self.baseline_mock.get_anomaly_score.return_value = -0.1
        
        self.predictor = IndustrialFailurePredictor(self.config, self.baseline_mock)
        
    def test_top_influencers_with_empty_shap(self):
        """Test: Top influencers funciona cuando SHAP está vacío (CORRECCIÓN #1)."""
        recent_data = pd.DataFrame({
            'var1': [10.0, 11.0, 12.0, 13.0, 14.0]
        })
        
        # Simular que XAI no tiene explicaciones
        self.predictor.xai_explainer.explain_current_risk = MagicMock(return_value=[])
        
        result = self.predictor.predict(recent_data)
        
        # Debe generar top_influencers sin fallar
        self.assertIn('top_influencers', result)
        self.assertIsInstance(result['top_influencers'], list)
        self.assertGreater(len(result['top_influencers']), 0)
        
        # Verificar formato: (var, risk, explanation)
        for item in result['top_influencers']:
            self.assertEqual(len(item), 3)
            var, risk, expl = item
            self.assertIsInstance(var, str)
            self.assertIsInstance(risk, float)
            self.assertIsInstance(expl, str)
            
    def test_top_influencers_with_shap(self):
        """Test: Top influencers combina SHAP y riesgos correctamente."""
        recent_data = pd.DataFrame({
            'var1': [10.0, 11.0, 12.0, 13.0, 14.0]
        })
        
        # Simular explicaciones SHAP
        shap_explanations = [
            ('var1', 0.15, 'Aumenta riesgo en 15.0%'),
            ('var2', 0.10, 'Aumenta riesgo en 10.0%')
        ]
        self.predictor.xai_explainer.explain_current_risk = MagicMock(return_value=shap_explanations)
        
        result = self.predictor.predict(recent_data)
        
        # Debe combinar correctamente
        self.assertIn('top_influencers', result)
        self.assertGreater(len(result['top_influencers']), 0)
        
    def test_get_forecast_with_nan(self):
        """Test: Manejo robusto cuando get_forecast retorna NaN (CORRECCIÓN #3)."""
        # Simular que get_forecast retorna NaN
        self.baseline_mock.get_forecast.return_value = np.nan
        
        recent_data = pd.DataFrame({
            'var1': [10.0, 11.0, 12.0]
        })
        
        # No debe fallar
        result = self.predictor.predict(recent_data)
        self.assertIsNotNone(result)
        self.assertIn('variable_risks', result)
        
    def test_get_forecast_with_empty_series(self):
        """Test: Manejo robusto cuando recent_window está vacío."""
        recent_data = pd.DataFrame({
            'var1': []  # Serie vacía
        })
        
        # No debe fallar
        try:
            result = self.predictor.predict(recent_data)
            # Si no hay datos, puede retornar resultado vacío pero no debe fallar
            self.assertIsNotNone(result)
        except Exception as e:
            self.fail(f"Predictor falló con serie vacía: {e}")
            
    def test_ttf_estimation_increasing_trend(self):
        """Test: TTF estimation para tendencia ascendente."""
        # Datos con tendencia ascendente clara
        recent_data = pd.DataFrame({
            'var1': [10.0, 11.0, 12.0, 13.0, 14.0, 15.0]  # Tendencia +1 por punto
        })
        
        # Simular riesgo alto para activar TTF
        self.predictor.predictions['variable_risks'] = {'var1': 0.8}
        
        ttf_info = self.predictor._estimate_time_to_failure('var1', recent_data['var1'], 0.8)
        
        # Debe calcular TTF
        self.assertIn('ttf_hours', ttf_info)
        self.assertIn('confidence', ttf_info)
        self.assertIn('trajectory', ttf_info)
        
        # Para tendencia ascendente, trajectory debe ser 'increasing'
        self.assertEqual(ttf_info['trajectory'], 'increasing')
        
        # TTF debe ser un número positivo o None
        if ttf_info['ttf_hours'] is not None:
            self.assertGreaterEqual(ttf_info['ttf_hours'], 0)
            
    def test_ttf_estimation_stable_trend(self):
        """Test: TTF estimation para tendencia estable (sin cambio)."""
        # Datos estables
        recent_data = pd.DataFrame({
            'var1': [10.0, 10.0, 10.0, 10.0, 10.0]
        })
        
        ttf_info = self.predictor._estimate_time_to_failure('var1', recent_data['var1'], 0.3)
        
        # Para tendencia estable, TTF debe ser None
        self.assertIsNone(ttf_info['ttf_hours'])
        self.assertEqual(ttf_info['trajectory'], 'stable')
        
    def test_ttf_in_alerts(self):
        """Test: TTF se incluye en alertas cuando está disponible."""
        recent_data = pd.DataFrame({
            'var1': [10.0, 11.0, 12.0, 13.0, 14.0, 15.0]
        })
        
        # Simular riesgo alto para generar alerta
        self.predictor.predictions['variable_risks'] = {'var1': 0.8}
        self.predictor.xai_explainer.explain_current_risk = MagicMock(return_value=[])
        
        result = self.predictor.predict(recent_data)
        
        # Verificar que las alertas incluyen TTF si está disponible
        if result.get('alerts'):
            for alert in result['alerts']:
                if alert.get('ttf_info'):
                    self.assertIn('ttf_hours', alert['ttf_info'])
                    
    def test_variable_risk_calculation_robust(self):
        """Test: Cálculo de riesgo es robusto ante edge cases."""
        # Caso 1: Serie con un solo valor
        recent_data = pd.DataFrame({
            'var1': [10.0]
        })
        
        risk = self.predictor._calculate_variable_risk('var1', recent_data['var1'])
        self.assertIsInstance(risk, float)
        self.assertGreaterEqual(risk, 0.0)
        self.assertLessEqual(risk, 1.0)
        
        # Caso 2: Serie con valores constantes
        recent_data = pd.DataFrame({
            'var1': [10.0, 10.0, 10.0, 10.0]
        })
        
        risk = self.predictor._calculate_variable_risk('var1', recent_data['var1'])
        self.assertIsInstance(risk, float)


if __name__ == '__main__':
    unittest.main()
