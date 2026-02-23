"""
Tests unitarios para XAIExplainer
Valida que SHAP funcione correctamente y genere explicaciones válidas.
"""

import unittest
import numpy as np
import pandas as pd
from src.models.xai_explainer import XAIExplainer


class TestXAIExplainer(unittest.TestCase):
    """Tests para el módulo de explicabilidad XAI."""
    
    def setUp(self):
        """Configuración inicial para cada test."""
        self.explainer = XAIExplainer()
        
    def test_update_training_buffer(self):
        """Test: Buffer de entrenamiento se actualiza correctamente."""
        var_risks = {'var1': 0.5, 'var2': 0.7, 'var3': 0.3}
        recent_data = pd.DataFrame({
            'var1': [1.0, 1.1, 1.2],
            'var2': [2.0, 2.1, 2.2],
            'var3': [3.0, 3.1, 3.2]
        })
        
        self.explainer.update_training_buffer(var_risks, recent_data)
        
        # Verificar que se agregó un ejemplo
        self.assertEqual(len(self.explainer.training_data), 1)
        self.assertEqual(len(self.explainer.training_targets), 1)
        self.assertEqual(len(self.explainer.feature_names), 3)
        
        # Verificar que el target es el promedio de riesgos
        expected_target = np.mean([0.5, 0.7, 0.3])
        self.assertAlmostEqual(self.explainer.training_targets[0], expected_target, places=5)
        
    def test_training_buffer_window_limit(self):
        """Test: Buffer mantiene ventana deslizante de 500 ejemplos."""
        var_risks = {'var1': 0.5, 'var2': 0.6}
        recent_data = pd.DataFrame({'var1': [1.0], 'var2': [2.0]})
        
        # Agregar 600 ejemplos
        for i in range(600):
            self.explainer.update_training_buffer(var_risks, recent_data)
        
        # Debe mantener solo los últimos 500
        self.assertEqual(len(self.explainer.training_data), 500)
        self.assertEqual(len(self.explainer.training_targets), 500)
        
    def test_train_surrogate_model_insufficient_data(self):
        """Test: No entrena con menos de 50 ejemplos."""
        var_risks = {'var1': 0.5}
        recent_data = pd.DataFrame({'var1': [1.0]})
        
        # Agregar solo 10 ejemplos
        for i in range(10):
            self.explainer.update_training_buffer(var_risks, recent_data)
        
        result = self.explainer.train_surrogate_model()
        self.assertFalse(result)
        self.assertIsNone(self.explainer.surrogate_model)
        
    def test_train_surrogate_model_success(self):
        """Test: Entrena modelo surrogate con suficientes datos."""
        var_risks = {'var1': 0.5, 'var2': 0.6, 'var3': 0.4}
        recent_data = pd.DataFrame({
            'var1': np.random.randn(100),
            'var2': np.random.randn(100),
            'var3': np.random.randn(100)
        })
        
        # Agregar 100 ejemplos
        for i in range(100):
            var_risks_cycle = {k: v + np.random.randn() * 0.1 for k, v in var_risks.items()}
            self.explainer.update_training_buffer(var_risks_cycle, recent_data.iloc[[i]])
        
        result = self.explainer.train_surrogate_model()
        self.assertTrue(result)
        self.assertIsNotNone(self.explainer.surrogate_model)
        self.assertIsNotNone(self.explainer.explainer)
        
    def test_explain_current_risk_without_model(self):
        """Test: Genera fallback cuando no hay modelo entrenado."""
        current_variables = {'var1': 0.5, 'var2': 0.6}
        
        explanations = self.explainer.explain_current_risk(current_variables)
        
        # Debe retornar lista vacía (no hay modelo)
        self.assertEqual(len(explanations), 0)
        
    def test_explain_current_risk_with_model(self):
        """Test: Genera explicaciones SHAP cuando hay modelo entrenado."""
        var_risks = {'var1': 0.5, 'var2': 0.6, 'var3': 0.4}
        recent_data = pd.DataFrame({
            'var1': np.random.randn(100),
            'var2': np.random.randn(100),
            'var3': np.random.randn(100)
        })
        
        # Entrenar modelo
        for i in range(100):
            var_risks_cycle = {k: v + np.random.randn() * 0.1 for k, v in var_risks.items()}
            self.explainer.update_training_buffer(var_risks_cycle, recent_data.iloc[[i]])
        
        self.explainer.train_surrogate_model()
        
        # Generar explicaciones
        current_variables = {'var1': 0.5, 'var2': 0.6, 'var3': 0.4}
        explanations = self.explainer.explain_current_risk(current_variables)
        
        # Debe retornar explicaciones para todas las variables
        self.assertGreater(len(explanations), 0)
        self.assertEqual(len(explanations), 3)  # 3 variables
        
        # Verificar formato: (variable, shap_value, interpretación)
        for var, shap_val, interpretation in explanations:
            self.assertIsInstance(var, str)
            self.assertIsInstance(shap_val, float)
            self.assertIsInstance(interpretation, str)
            
    def test_explanations_sorted_by_impact(self):
        """Test: Explicaciones están ordenadas por impacto absoluto."""
        var_risks = {'var1': 0.5, 'var2': 0.6, 'var3': 0.4}
        recent_data = pd.DataFrame({
            'var1': np.random.randn(100),
            'var2': np.random.randn(100),
            'var3': np.random.randn(100)
        })
        
        # Entrenar modelo
        for i in range(100):
            var_risks_cycle = {k: v + np.random.randn() * 0.1 for k, v in var_risks.items()}
            self.explainer.update_training_buffer(var_risks_cycle, recent_data.iloc[[i]])
        
        self.explainer.train_surrogate_model()
        
        explanations = self.explainer.explain_current_risk(var_risks)
        
        # Verificar que están ordenadas por impacto absoluto (descendente)
        if len(explanations) > 1:
            shap_values = [abs(shap_val) for _, shap_val, _ in explanations]
            self.assertEqual(shap_values, sorted(shap_values, reverse=True))
            
    def test_fallback_explanation(self):
        """Test: Fallback usa feature importance cuando SHAP falla."""
        var_risks = {'var1': 0.5, 'var2': 0.6}
        recent_data = pd.DataFrame({
            'var1': np.random.randn(100),
            'var2': np.random.randn(100)
        })
        
        # Entrenar modelo
        for i in range(100):
            var_risks_cycle = {k: v + np.random.randn() * 0.1 for k, v in var_risks.items()}
            self.explainer.update_training_buffer(var_risks_cycle, recent_data.iloc[[i]])
        
        self.explainer.train_surrogate_model()
        
        # Simular fallo de SHAP (explicador None)
        self.explainer.explainer = None
        
        explanations = self.explainer._fallback_explanation(var_risks)
        
        # Debe usar feature importance
        self.assertGreater(len(explanations), 0)
        for var, importance, interpretation in explanations:
            self.assertIsInstance(importance, float)
            self.assertGreaterEqual(importance, 0.0)


if __name__ == '__main__':
    unittest.main()
