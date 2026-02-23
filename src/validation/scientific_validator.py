"""
✅ FASE 2 P1: Framework de Validación Científica
Valida modelos con rigor estadístico y métricas industriales.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class ScientificValidator:
    """
    Validador científico para modelos de prognosis industrial.
    Implementa validación cruzada temporal, tests estadísticos y métricas específicas.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.validation_history = []
        
    def validate_forecast_model(self, 
                                model, 
                                test_data: pd.Series, 
                                forecast_horizon: int = 10) -> Dict[str, Any]:
        """
        Valida un modelo de forecasting usando validación cruzada temporal.
        Args:
            model: Modelo entrenado (SARIMAX o Prophet)
            test_data: Serie de prueba
            forecast_horizon: Horizonte de predicción
        Returns:
            Dict con métricas de performance
        """
        try:
            predictions = []
            actuals = []
            
            # Validación cruzada temporal (rolling window)
            for i in range(len(test_data) - forecast_horizon):
                # Predecir próximos valores
                if hasattr(model, 'forecast'):  # SARIMAX
                    pred = model.forecast(steps=forecast_horizon)
                elif hasattr(model, 'predict'):  # Prophet
                    # ✅ Timestamp-driven: construir futuro desde el índice real de test_data
                    if not isinstance(test_data.index, pd.DatetimeIndex):
                        return {'error': 'Prophet requiere test_data con DatetimeIndex para validación timestamp-driven'}

                    idx = test_data.index.dropna().sort_values()
                    if len(idx) < 3:
                        return {'error': 'Insuficientes timestamps para inferir dt en validación Prophet'}

                    diffs = idx.to_series().diff().dropna().dt.total_seconds()
                    diffs = diffs[diffs > 0]
                    if diffs.empty:
                        return {'error': 'No se pudo inferir dt (diffs vacíos) en validación Prophet'}

                    dt_seconds = float(diffs.median())
                    last_ts = pd.to_datetime(idx[-1])
                    future = pd.DataFrame({
                        'ds': [last_ts + pd.Timedelta(seconds=dt_seconds * (j + 1)) for j in range(forecast_horizon)]
                    })
                    pred = model.predict(future)['yhat'].values
                else:
                    return {'error': 'Modelo no soportado'}
                
                # Valores reales
                actual = test_data.iloc[i:i+forecast_horizon].values
                
                if len(pred) == len(actual):
                    predictions.extend(pred)
                    actuals.extend(actual)
            
            if len(predictions) == 0:
                return {'error': 'No se pudieron generar predicciones'}
            
            # Calcular métricas
            rmse = np.sqrt(mean_squared_error(actuals, predictions))
            mae = mean_absolute_error(actuals, predictions)
            r2 = r2_score(actuals, predictions)
            
            # MAPE (Mean Absolute Percentage Error)
            mape = np.mean(np.abs((np.array(actuals) - np.array(predictions)) / (np.array(actuals) + 1e-10))) * 100
            
            # Test de Diebold-Mariano (comparación con baseline naive)
            naive_forecast = [test_data.iloc[i] for i in range(len(predictions))]
            dm_stat, dm_pvalue = self._diebold_mariano_test(actuals, predictions, naive_forecast)
            
            metrics = {
                'rmse': float(rmse),
                'mae': float(mae),
                'r2': float(r2),
                'mape': float(mape),
                'diebold_mariano_stat': float(dm_stat),
                'diebold_mariano_pvalue': float(dm_pvalue),
                'is_better_than_naive': dm_pvalue < 0.05 and dm_stat < 0
            }
            
            self.logger.info(f"✅ Validación completada: RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}")
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error en validación de modelo: {e}")
            return {'error': str(e)}
    
    def _diebold_mariano_test(self, 
                              actual: List[float], 
                              forecast1: List[float], 
                              forecast2: List[float]) -> Tuple[float, float]:
        """
        Test de Diebold-Mariano para comparar accuracy de dos forecasts.
        H0: Los dos modelos tienen igual accuracy
        Returns: (DM statistic, p-value)
        """
        try:
            e1 = np.array(actual) - np.array(forecast1)
            e2 = np.array(actual) - np.array(forecast2)
            
            d = (e1 ** 2) - (e2 ** 2)
            
            mean_d = np.mean(d)
            var_d = np.var(d, ddof=1)
            n = len(d)
            
            dm_stat = mean_d / np.sqrt(var_d / n)
            
            # P-value (two-tailed test)
            pvalue = 2 * (1 - stats.norm.cdf(abs(dm_stat)))
            
            return dm_stat, pvalue
            
        except Exception as e:
            self.logger.warning(f"Error en Diebold-Mariano test: {e}")
            return 0.0, 1.0
    
    def validate_anomaly_detector(self, 
                                  detector, 
                                  test_data: np.ndarray, 
                                  true_labels: np.ndarray = None) -> Dict[str, Any]:
        """
        Valida un detector de anomalías (Isolation Forest).
        Args:
            detector: Modelo Isolation Forest entrenado
            test_data: Datos de prueba
            true_labels: Etiquetas reales (1=normal, -1=anomalía) si están disponibles
        Returns:
            Dict con métricas de detección
        """
        try:
            predictions = detector.predict(test_data.reshape(-1, 1))
            scores = detector.decision_function(test_data.reshape(-1, 1))
            
            metrics = {
                'n_anomalies_detected': int(np.sum(predictions == -1)),
                'anomaly_rate': float(np.mean(predictions == -1)),
                'mean_score': float(np.mean(scores)),
                'std_score': float(np.std(scores))
            }
            
            # Si tenemos true labels, calcular precision/recall
            if true_labels is not None and len(true_labels) == len(predictions):
                from sklearn.metrics import precision_score, recall_score, f1_score
                
                metrics['precision'] = float(precision_score(true_labels, predictions, pos_label=-1, zero_division=0))
                metrics['recall'] = float(recall_score(true_labels, predictions, pos_label=-1, zero_division=0))
                metrics['f1_score'] = float(f1_score(true_labels, predictions, pos_label=-1, zero_division=0))
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error en validación de detector de anomalías: {e}")
            return {'error': str(e)}
    
    def validate_baseline_stability(self, 
                                    baseline_history: List[Dict[str, float]], 
                                    window_size: int = 10) -> Dict[str, Any]:
        """
        Valida la estabilidad del baseline a lo largo del tiempo.
        Args:
            baseline_history: Lista de estadísticas del baseline en el tiempo
            window_size: Tamaño de ventana para análisis de estabilidad
        Returns:
            Dict con métricas de estabilidad
        """
        if len(baseline_history) < window_size:
            return {'stability_score': 1.0, 'is_stable': True, 'trend': 'insufficient_data'}
        
        try:
            # Extraer mediana y IQR de la historia
            medians = [h.get('median', 0) for h in baseline_history[-window_size:]]
            iqrs = [h.get('iqr', 0) for h in baseline_history[-window_size:]]
            
            # Calcular coeficiente de variación (CV)
            cv_median = np.std(medians) / (np.mean(medians) + 1e-10)
            cv_iqr = np.std(iqrs) / (np.mean(iqrs) + 1e-10)
            
            # Test de Mann-Kendall para detectar tendencia
            tau, trend_pvalue = stats.kendalltau(range(len(medians)), medians)
            
            # Criterio de estabilidad: CV < 0.3 y no hay tendencia significativa
            is_stable = (cv_median < 0.3 and cv_iqr < 0.3 and trend_pvalue > 0.05)
            
            stability_score = 1.0 - min(1.0, cv_median + cv_iqr) / 2
            
            trend = 'stable'
            if trend_pvalue < 0.05:
                trend = 'increasing' if tau > 0 else 'decreasing'
            
            return {
                'stability_score': float(stability_score),
                'is_stable': bool(is_stable),
                'cv_median': float(cv_median),
                'cv_iqr': float(cv_iqr),
                'trend': trend,
                'trend_pvalue': float(trend_pvalue)
            }
            
        except Exception as e:
            self.logger.error(f"Error en validación de estabilidad: {e}")
            return {'error': str(e)}
    
    def generate_validation_report(self, 
                                   model_metrics: Dict[str, Dict[str, Any]], 
                                   stability_metrics: Dict[str, Any]) -> str:
        """
        Genera un reporte de validación científica en formato texto.
        """
        report = []
        report.append("=" * 80)
        report.append("REPORTE DE VALIDACIÓN CIENTÍFICA - PROGNOSIS II")
        report.append("=" * 80)
        report.append("")
        
        # Sección: Métricas de Modelos
        report.append("1. MÉTRICAS DE PERFORMANCE DE MODELOS")
        report.append("-" * 80)
        for model_name, metrics in model_metrics.items():
            report.append(f"\n{model_name.upper()}:")
            for metric, value in metrics.items():
                if isinstance(value, float):
                    report.append(f"  - {metric}: {value:.4f}")
                else:
                    report.append(f"  - {metric}: {value}")
        report.append("")
        
        # Sección: Estabilidad del Baseline
        report.append("2. ESTABILIDAD DEL BASELINE")
        report.append("-" * 80)
        for metric, value in stability_metrics.items():
            if isinstance(value, float):
                report.append(f"  - {metric}: {value:.4f}")
            else:
                report.append(f"  - {metric}: {value}")
        report.append("")
        
        # Sección: Recomendaciones
        report.append("3. RECOMENDACIONES")
        report.append("-" * 80)
        
        # Análisis de R²
        for model_name, metrics in model_metrics.items():
            if 'r2' in metrics:
                r2 = metrics['r2']
                if r2 < 0.5:
                    report.append(f"  ⚠️ {model_name}: R² bajo ({r2:.2f}). Considerar re-entrenamiento.")
                elif r2 > 0.8:
                    report.append(f"  ✅ {model_name}: Excelente R² ({r2:.2f}).")
        
        # Análisis de estabilidad
        if not stability_metrics.get('is_stable', True):
            report.append(f"  ⚠️ Baseline inestable. Revisar concept drift.")
        else:
            report.append(f"  ✅ Baseline estable.")
        
        report.append("")
        report.append("=" * 80)
        
        return "\n".join(report)
