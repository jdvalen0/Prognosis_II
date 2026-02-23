import unittest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock
from src.config import SystemConfig
from src.models.baseline_modeler import BaselineModeler
from src.models.predictor import IndustrialFailurePredictor

class TestEnsembleIntegration(unittest.TestCase):
    
    def setUp(self):
        self.config = SystemConfig()
        # Reducir umbral para que el ensamble entrene modelos en tests pequeños
        self.config.MODEL_PARAMS['min_data_points'] = 50
        # Mock database manager to avoid connection errors in unit tests
        self.db = MagicMock()
        # No es necesario configurar el engine del mock a menos que se llame
        self.modeler = BaselineModeler(self.config, self.db)
        
    def test_residual_calculation(self):
        # Crear un escenario donde el valor actual coincide perfectamente con el pronóstico
        variable = "sensor_test"
        self.modeler.results['adaptive_limits'][variable] = {
            'baseline': 100.0, 'upper': 110.0, 'lower': 90.0
        }
        
        # Mock de datos históricos
        data = pd.Series([100.0]*10, name=variable)
        
        # Sin modelos, el forecast es el baseline (100.0)
        predictor = IndustrialFailurePredictor(self.config, self.modeler)
        prob = predictor._calculate_variable_risk(variable, data)
        
        # Si el valor es 100 y el forecast es 100, el residuo es 0.
        self.assertAlmostEqual(prob, 0.0, places=7)
        print(f"Validado: Riesgo casi cero ({prob:.2e}) cuando el residuo es casi cero.")

    def test_isolation_forest_impact(self):
        variable = "anomaly_test"
        self.modeler.results['adaptive_limits'][variable] = {
            'baseline': 50.0, 'upper': 60.0, 'lower': 40.0
        }
        
        # Entrenar un Isolation Forest real con valores alrededor de 50
        train_data = pd.Series(np.random.normal(50, 1, 100), name=variable)
        results = self.modeler.fit_ensemble(variable, train_data)
        
        # Evitar persistencia DB: inyectar modelos y límites en memoria
        if 'models' in results:
            self.modeler.models[variable] = results['models']
        self.modeler.results['adaptive_limits'][variable] = results['limits']
        
        predictor = IndustrialFailurePredictor(self.config, self.modeler)
        
        # Valor normal (50)
        prob_normal = predictor._calculate_variable_risk(variable, pd.Series([50.0]*5))
        
        # Valor anómalo (500)
        prob_anomaly = predictor._calculate_variable_risk(variable, pd.Series([500.0]*5))
        
        iso_score = self.modeler.get_anomaly_score(variable, 500.0)
        print(f"DEBUG: iso_score para 500.0 = {iso_score}")
        
        self.assertTrue(prob_anomaly > prob_normal)
        self.assertTrue(prob_anomaly >= 0.4 or iso_score < 0) 
        print(f"Validado: Riesgo anómalo ({prob_anomaly:.2f}) > Riesgo normal ({prob_normal:.2f})")

if __name__ == '__main__':
    unittest.main()
