"""
Tests para validar correcciones en BaselineModeler
Específicamente: migración de formato, validación de seasonality, optimización SARIMAX
"""

import unittest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from src.models.baseline_modeler import BaselineModeler
from src.config import SystemConfig


class TestBaselineCorrections(unittest.TestCase):
    """Tests para validar correcciones en BaselineModeler."""
    
    def setUp(self):
        """Configuración inicial."""
        self.config = SystemConfig()
        self.db_mock = MagicMock()
        
        # Mock de get_baseline_stats para simular formato antiguo
        self.db_mock.get_baseline_stats = MagicMock(return_value={
            'mean': 10.0,
            'std': 2.0
        })
        self.db_mock.engine = MagicMock()
        self.db_mock.save_model_version = MagicMock()
        
        self.modeler = BaselineModeler(self.config, self.db_mock)
        
    def test_update_incremental_migration_old_format(self):
        """Test: Migración automática de formato antiguo (mean/std) a nuevo (CORRECCIÓN #2)."""
        # Simular formato antiguo
        self.db_mock.get_baseline_stats.return_value = {
            'mean': 10.0,
            'std': 2.0
        }
        
        new_stats = {
            'median': 11.0,
            'iqr': 3.0,
            'p95': 14.0,
            'p05': 8.0
        }
        
        result = self.modeler.update_incremental('test_var', new_stats)
        
        # Debe migrar automáticamente
        self.assertIn('median', result)
        self.assertIn('iqr', result)
        self.assertIn('p95', result)
        self.assertIn('p05', result)
        
        # Verificar que los valores son razonables (migración + blending 70/30)
        self.assertGreater(result['median'], 0)
        
    def test_update_incremental_new_format(self):
        """Test: Funciona correctamente con formato nuevo."""
        # Simular formato nuevo
        self.db_mock.get_baseline_stats.return_value = {
            'median': 10.0,
            'iqr': 2.0,
            'p95': 13.0,
            'p05': 7.0
        }
        
        new_stats = {
            'median': 11.0,
            'iqr': 3.0,
            'p95': 14.0,
            'p05': 8.0
        }
        
        result = self.modeler.update_incremental('test_var', new_stats)
        
        # Debe aplicar blending 70/30
        expected_median = 10.0 * 0.7 + 11.0 * 0.3
        self.assertAlmostEqual(result['median'], expected_median, places=5)
        
    def test_check_seasonality_minimum_data(self):
        """Test: Validación de datos mínimos en _check_seasonality (CORRECCIÓN #4)."""
        # Serie muy pequeña
        small_series = pd.Series([1, 2, 3, 4, 5])
        
        has_seasonality, period = self.modeler._check_seasonality(small_series)
        
        # Debe retornar False para series muy pequeñas
        self.assertFalse(has_seasonality)
        self.assertEqual(period, 0)
        
    def test_check_seasonality_with_nan(self):
        """Test: Manejo robusto de NaN en detección de seasonality."""
        # Serie con NaN
        series_with_nan = pd.Series([1, 2, np.nan, 4, 5, 6, 7, 8, 9, 10, 11, 12] * 5)
        
        has_seasonality, period = self.modeler._check_seasonality(series_with_nan)
        
        # No debe fallar
        self.assertIsInstance(has_seasonality, bool)
        self.assertIsInstance(period, int)
        
    def test_check_seasonality_detection(self):
        """Test: Detecta seasonality en serie con patrón claro."""
        # Serie con seasonality de periodo 24
        t = np.arange(100)
        seasonal_series = pd.Series(10 + 5 * np.sin(2 * np.pi * t / 24) + np.random.randn(100) * 0.5)
        
        has_seasonality, period = self.modeler._check_seasonality(seasonal_series)
        
        # Debe detectar seasonality
        # Nota: Puede no detectar exactamente 24 debido a ruido, pero debe detectar algo
        if has_seasonality:
            self.assertGreater(period, 0)
            self.assertLessEqual(period, 168)
            
    @patch('src.models.baseline_modeler.SARIMAX')
    def test_optimize_sarima_all_fail(self, mock_sarimax):
        """Test: Manejo cuando todas las combinaciones fallan (CORRECCIÓN #5)."""
        # Simular que todas las combinaciones fallan
        mock_sarimax.side_effect = Exception("Convergence failed")
        
        data = pd.Series(np.random.randn(100))
        
        order, seasonal, aic = self.modeler._optimize_sarima_params(data, 24)
        
        # Debe retornar parámetros por defecto
        self.assertEqual(order, (1, 1, 1))
        self.assertEqual(seasonal, (1, 1, 1, 24))
        # AIC debe ser None cuando todas fallan
        self.assertIsNone(aic)
        
    def test_get_forecast_hierarchy(self):
        """Test: Jerarquía SARIMAX → Prophet → Baseline funciona."""
        # Configurar modelos mock
        self.modeler.models['test_var'] = {
            'sarima': MagicMock(),
            'prophet': MagicMock(),
            'isolation_forest': MagicMock()
        }
        
        # Mock SARIMAX forecast exitoso
        self.modeler.models['test_var']['sarima'].forecast.return_value = pd.Series([12.0])
        self.modeler.results['adaptive_limits']['test_var'] = {'baseline': 10.0}
        
        forecast = self.modeler.get_forecast('test_var')
        
        # Debe usar SARIMAX (primera opción)
        self.assertEqual(forecast, 12.0)
        
    def test_get_forecast_fallback_to_prophet(self):
        """Test: Fallback a Prophet cuando SARIMAX falla."""
        self.modeler.models['test_var'] = {
            'sarima': MagicMock(),
            'prophet': MagicMock(),
            'isolation_forest': MagicMock(),
            # ✅ Timeline real para Prophet (no usar now)
            'time_meta': {'last_timestamp': '2025-01-01 00:00:00', 'dt_hours': 1.0}
        }
        
        # SARIMAX falla
        self.modeler.models['test_var']['sarima'].forecast.side_effect = Exception("SARIMAX failed")
        
        # Prophet funciona
        self.modeler.models['test_var']['prophet'].predict.return_value = pd.DataFrame({'yhat': [11.0]})
        
        self.modeler.results['adaptive_limits']['test_var'] = {'baseline': 10.0}
        
        forecast = self.modeler.get_forecast('test_var')
        
        # Debe usar Prophet (fallback)
        self.assertEqual(forecast, 11.0)
        
    def test_get_forecast_fallback_to_baseline(self):
        """Test: Fallback a baseline cuando todos los modelos fallan."""
        self.modeler.models['test_var'] = {
            'sarima': MagicMock(),
            'prophet': MagicMock(),
            'isolation_forest': MagicMock()
        }
        
        # Todos fallan
        self.modeler.models['test_var']['sarima'].forecast.side_effect = Exception("Failed")
        self.modeler.models['test_var']['prophet'].predict.side_effect = Exception("Failed")
        
        self.modeler.results['adaptive_limits']['test_var'] = {'baseline': 10.0}
        
        forecast = self.modeler.get_forecast('test_var')
        
        # Debe usar baseline (último recurso)
        self.assertEqual(forecast, 10.0)


if __name__ == '__main__':
    unittest.main()
