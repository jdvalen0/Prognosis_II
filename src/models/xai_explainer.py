"""
✅ FASE 2 P1: Módulo de Explicabilidad (XAI) usando SHAP
Proporciona interpretabilidad científica para diagnóstico de fallas.
"""

import logging
import numpy as np
import pandas as pd
import shap
from typing import Dict, List, Any, Tuple
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


class XAIExplainer:
    """
    Explainer avanzado usando SHAP para identificar variables influyentes.
    Estrategia: Entrenar modelo surrogate (RF/GBM) que aprende el mapeo 
    variables→riesgo, luego aplicar SHAP para descomponer contribuciones.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.surrogate_model = None
        self.explainer = None
        self.feature_names = []
        self.training_data = []
        self.training_targets = []
        
    def update_training_buffer(self, variable_risks: Dict[str, float], recent_data: pd.DataFrame):
        """
        Acumula ejemplos de entrenamiento para el modelo surrogate.
        Args:
            variable_risks: Dict con riesgos calculados por variable
            recent_data: DataFrame con valores actuales de las variables
        """
        if recent_data is None or getattr(recent_data, "empty", False):
            # No hay datos para extraer features
            return

        # Extraer valores de las variables presentes en variable_risks
        features = []
        for var in variable_risks.keys():
            if var in recent_data.columns:
                # Usar el último valor disponible
                try:
                    features.append(recent_data[var].iloc[-1])
                except Exception:
                    features.append(0.0)
            else:
                features.append(0.0)
        
        # Target: riesgo promedio del sistema (lo que queremos explicar)
        target = np.mean(list(variable_risks.values()))
        
        if len(features) > 0:
            self.training_data.append(features)
            self.training_targets.append(target)
            self.feature_names = list(variable_risks.keys())
            
            # Mantener ventana deslizante (últimos 500 ejemplos)
            if len(self.training_data) > 500:
                self.training_data = self.training_data[-500:]
                self.training_targets = self.training_targets[-500:]
    
    def train_surrogate_model(self) -> bool:
        """
        Entrena modelo surrogate para aprender la función riesgo = f(variables).
        Returns: True si el entrenamiento fue exitoso
        """
        if len(self.training_data) < 50:  # Mínimo 50 ejemplos para entrenar
            # No mostrar warning si es la primera vez (comportamiento esperado)
            if len(self.training_data) > 0:
                self.logger.debug(f"XAI: Acumulando ejemplos ({len(self.training_data)}/50). El explainer se entrenará automáticamente cuando haya suficientes datos.")
            return False
        
        try:
            X = np.array(self.training_data)
            y = np.array(self.training_targets)
            
            # Usar Gradient Boosting por su capacidad de capturar interacciones complejas
            self.surrogate_model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                random_state=42
            )
            self.surrogate_model.fit(X, y)
            
            # Crear explainer SHAP
            self.explainer = shap.Explainer(self.surrogate_model, X)
            
            self.logger.info(f"✅ Modelo surrogate XAI entrenado con {len(X)} ejemplos")
            return True
            
        except Exception as e:
            self.logger.error(f"Error entrenando modelo surrogate XAI: {e}")
            return False
    
    def explain_current_risk(self, current_variables: Dict[str, float]) -> List[Tuple[str, float, str]]:
        """
        Genera explicaciones SHAP para el riesgo actual.
        Args:
            current_variables: Dict con valores actuales de las variables
        Returns:
            Lista de (variable, shap_value, interpretación) ordenada por impacto
        """
        if self.explainer is None or self.surrogate_model is None:
            # Intentar entrenar silenciosamente si hay suficientes datos
            if not self.train_surrogate_model():
                # Si no hay suficientes datos, retornar explicación basada en riesgo simple
                return self._fallback_explanation(current_variables)
        
        try:
            # Preparar datos de entrada (mismo orden que training)
            X_explain = []
            for var in self.feature_names:
                X_explain.append(current_variables.get(var, 0.0))
            
            X_explain = np.array([X_explain])
            
            # Calcular SHAP values
            shap_values = self.explainer(X_explain)
            
            # Extraer contribuciones y generar explicaciones
            explanations = []
            for i, var in enumerate(self.feature_names):
                shap_val = float(shap_values.values[0][i])
                
                # Interpretar el SHAP value
                if abs(shap_val) < 0.01:
                    interpretation = "Impacto despreciable"
                elif shap_val > 0:
                    interpretation = f"Aumenta riesgo en {abs(shap_val):.1%}"
                else:
                    interpretation = f"Reduce riesgo en {abs(shap_val):.1%}"
                
                explanations.append((var, shap_val, interpretation))
            
            # Ordenar por impacto absoluto (descendente)
            explanations.sort(key=lambda x: abs(x[1]), reverse=True)
            
            return explanations
            
        except Exception as e:
            self.logger.error(f"Error generando explicaciones SHAP: {e}")
            return self._fallback_explanation(current_variables)
    
    def _fallback_explanation(self, current_variables: Dict[str, float]) -> List[Tuple[str, float, str]]:
        """
        Explicación de respaldo usando importancia de features del modelo surrogate.
        """
        if self.surrogate_model is None:
            return []
        
        try:
            feature_importance = self.surrogate_model.feature_importances_
            explanations = []
            
            for i, var in enumerate(self.feature_names):
                importance = float(feature_importance[i])
                interpretation = f"Importancia relativa: {importance:.1%}"
                explanations.append((var, importance, interpretation))
            
            explanations.sort(key=lambda x: abs(x[1]), reverse=True)
            return explanations
            
        except Exception:
            return []
    
    def get_global_feature_importance(self) -> Dict[str, float]:
        """
        Obtiene la importancia global de cada feature a través de todas las predicciones.
        Útil para reportes de auditoría y análisis de largo plazo.
        """
        if self.explainer is None or len(self.training_data) < 50:
            return {}
        
        try:
            X = np.array(self.training_data[-100:])  # Últimos 100 ejemplos
            shap_values = self.explainer(X)
            
            # Calcular importancia media absoluta
            mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
            
            importance_dict = {}
            for i, var in enumerate(self.feature_names):
                importance_dict[var] = float(mean_abs_shap[i])
            
            return importance_dict
            
        except Exception as e:
            self.logger.error(f"Error calculando importancia global: {e}")
            return {}
