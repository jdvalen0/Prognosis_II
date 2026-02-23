import logging
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Tuple
from src.config import SystemConfig
from src.models.xai_explainer import XAIExplainer

class IndustrialFailurePredictor:
    """
    ✅ FASE 2 P1: Motor de predicción con XAI integrado (SHAP).
    Cambios: Incluye explicabilidad científica real para diagnóstico de causas raíz.
    """
    
    def __init__(self, config: SystemConfig, baseline_learner):
        self.config = config
        self.baseline = baseline_learner
        self.xai_explainer = XAIExplainer()  # ✅ Integración XAI
        self.predictions = {
            'current_risk': {},
            'variable_risks': {},  # Inicializado para evitar KeyError en EMA
            'alerts': [],
            'system_health': {'probability': 0.0, 'status': 'normal'}
        }
        self._setup_logging()

    def _setup_logging(self):
        self.logger = logging.getLogger(__name__)

    def _infer_dt_hours(self, recent_data: pd.DataFrame) -> float:
        """
        Inferir delta-t (en horas) desde timestamp real (columna o índice).
        ✅ Agnosticismo total: nunca asumir frecuencia.
        """
        try:
            # 1) Columna timestamp/fecha
            date_col = next((c for c in ['timestamp', 'fecha', 'datetime', 'date'] if c in recent_data.columns), None)
            if date_col:
                ts = pd.to_datetime(recent_data[date_col], errors='coerce').dropna().sort_values()
                if len(ts) >= 3:
                    diffs = ts.diff().dropna().dt.total_seconds()
                    diffs = diffs[diffs > 0]
                    if not diffs.empty:
                        return float(diffs.median() / 3600.0)

            # 2) Índice temporal
            if isinstance(recent_data.index, pd.DatetimeIndex):
                idx = recent_data.index.dropna().sort_values()
                if len(idx) >= 3:
                    diffs = idx.to_series().diff().dropna().dt.total_seconds()
                    diffs = diffs[diffs > 0]
                    if not diffs.empty:
                        return float(diffs.median() / 3600.0)
        except Exception:
            pass
        return 1.0

    def _calculate_variable_risk(self, variable: str, current_batch: pd.Series, dt_hours: float = 1.0) -> float:
        """
        ✅ CORRECCIÓN V9: Calcula probabilidad de falla usando la misma fórmula que el notebook.
        Alineado con prognosis_research_EM.pdf (Sección 3.2):
        - 40% desviación media absoluta normalizada
        - 40% proporción fuera de límites
        - 20% fuerza de tendencia
        
        Esta es la fórmula exacta del notebook Prognosis01 Set2.ipynb (Celda 5).
        """
        if variable not in self.baseline.results['adaptive_limits']:
            return 0.0
            
        limits = self.baseline.results['adaptive_limits'][variable]
        current_values = current_batch.values
        
        # Validar que hay datos
        if len(current_values) == 0:
            return 0.0
        
        baseline_mean = limits.get('baseline', 0.0)
        range_size = limits.get('upper', baseline_mean) - limits.get('lower', baseline_mean)
        
        if range_size <= 0:
            return 0.0
        
        # 1. ✅ DESVIACIÓN MEDIA ABSOLUTA NORMALIZADA (como notebook)
        # normalized_dev = (current_values - baseline_mean) / range_size
        normalized_dev = (current_values - baseline_mean) / range_size
        mean_deviation = float(np.mean(np.abs(normalized_dev)))
        deviation_factor = min(1.0, mean_deviation)
        
        # 2. ✅ PROPORCIÓN FUERA DE LÍMITES (como notebook)
        # out_of_limits = fracción de puntos que exceden los límites adaptativos
        out_of_limits = float(np.mean(
            (current_values > limits.get('upper', baseline_mean)) | 
            (current_values < limits.get('lower', baseline_mean))
        ))
        limit_factor = min(1.0, out_of_limits)
        
        # 3. ✅ FUERZA DE TENDENCIA (como notebook)
        # trend_strength = abs(slope) * len(values) / std(values)
        trend_factor = 0.0
        if len(current_values) >= 2:
            try:
                std_val = float(np.std(current_values))
                if std_val > 1e-12:
                    # Calcular pendiente (usando índice, no tiempo real para consistencia con notebook)
                    x = np.arange(len(current_values))
                    slope, _ = np.polyfit(x, current_values, 1)
                    if not np.isnan(slope):
                        # Fórmula exacta del notebook: abs(slope) * len(values) / std(values)
                        trend_strength = abs(float(slope)) * len(current_values) / std_val
                        trend_factor = min(1.0, trend_strength)
            except Exception:
                trend_factor = 0.0
        
        # ✅ FÓRMULA EXACTA DEL NOTEBOOK (0.4/0.4/0.2)
        probability = (
            0.4 * deviation_factor +
            0.4 * limit_factor +
            0.2 * trend_factor
        )
        
        # ✅ CORRECCIÓN V10: PARIDAD CON NOTEBOOK - NO usar suavizado temporal
        # El notebook retorna la probabilidad directamente, sin suavizado EMA
        # Esto asegura que cada ejecución calcule probabilidad fresca basada en datos actuales
        return float(probability)
    
    def _estimate_time_to_failure(self, variable: str, current_batch: pd.Series, current_risk: float, dt_hours: float = 1.0) -> Dict[str, Any]:
        """
        ✅ FASE 2 P1: Estima tiempo hasta falla usando extrapolación de tendencia.
        Estrategia:
        1. Detectar tendencia actual (slope)
        2. Calcular distancia a umbrales críticos
        3. Estimar TTF = distancia / velocidad_tendencia
        
        Returns:
            Dict con 'ttf_hours', 'confidence', 'trajectory'
        """
        if variable not in self.baseline.results['adaptive_limits']:
            return {'ttf_hours': None, 'confidence': 0.0, 'trajectory': 'unknown'}
        
        limits = self.baseline.results['adaptive_limits'][variable]
        
        # 1. Calcular tendencia actual (regresión lineal) usando tiempo real
        x = np.arange(len(current_batch), dtype=float) * float(dt_hours)
        try:
            if current_batch is None or len(current_batch.dropna()) < 2:
                return {'ttf_hours': None, 'confidence': 0.0, 'trajectory': 'stable'}
            if float(current_batch.std()) < 1e-12:
                return {'ttf_hours': None, 'confidence': 0.0, 'trajectory': 'stable'}
            slope, intercept = np.polyfit(x, current_batch.values, 1)
        except Exception:
            return {'ttf_hours': None, 'confidence': 0.0, 'trajectory': 'stable'}
        
        # 2. Determinar dirección y velocidad
        if abs(slope) < 1e-6:  # Tendencia prácticamente plana
            return {'ttf_hours': None, 'confidence': 0.0, 'trajectory': 'stable'}
        
        current_value = current_batch.iloc[-1]
        
        # 3. Calcular distancia a umbrales críticos
        if slope > 0:  # Tendencia ascendente
            target_threshold = limits.get('p99', limits['upper'])
            distance = target_threshold - current_value
            trajectory = 'increasing'
        else:  # Tendencia descendente
            target_threshold = limits.get('p01', limits['lower'])
            distance = current_value - target_threshold
            trajectory = 'decreasing'
        
        # 4. Estimar TTF (en horas) porque slope está en unidades/horas
        if distance <= 0:  # Ya superó el umbral
            return {'ttf_hours': 0, 'confidence': 0.9, 'trajectory': trajectory}
        
        ttf_hours = float(distance / abs(slope))  # horas reales (timestamp-driven)
        
        # 5. Calcular confianza (basada en R² de la regresión)
        try:
            from scipy.stats import linregress
            _, _, r_value, _, _ = linregress(x, current_batch.values)
            confidence = float(r_value ** 2)  # R²
        except Exception:
            confidence = 0.5
        
        # 6. Ajustar TTF según nivel de riesgo actual (mayor riesgo = TTF más corto)
        risk_adjustment = 1.0 - (current_risk * 0.5)  # Reducir TTF hasta 50% si riesgo es 100%
        ttf_hours *= risk_adjustment
        
        return {
            'ttf_hours': max(0, ttf_hours),
            'confidence': confidence,
            'trajectory': trajectory,
            'slope': float(slope)
        }

    def predict(self, recent_data: pd.DataFrame) -> Dict[str, Any]:
        """
        ✅ FASE 2 P1: Predicción global con XAI + TTF estimation.
        Cambios: 
        - Incluye diagnóstico XAI científico (SHAP values)
        - Estimación de Time-to-Failure para variables críticas
        """
        self.logger.info("Iniciando ciclo de predicción...")
        dt_hours = self._infer_dt_hours(recent_data)
        var_risks = {}
        var_ttf = {}  # ✅ Time-to-Failure por variable
        
        for var in recent_data.columns:
            if var in self.baseline.results['adaptive_limits']:
                risk = self._calculate_variable_risk(var, recent_data[var], dt_hours=dt_hours)
                var_risks[var] = risk
        
                # ✅ Calcular TTF si el riesgo es significativo (>20%)
                if risk > 0.2:
                    ttf_info = self._estimate_time_to_failure(var, recent_data[var], risk, dt_hours=dt_hours)
                    var_ttf[var] = ttf_info
        
        # ✅ Actualizar buffer de entrenamiento para XAI
        self.xai_explainer.update_training_buffer(var_risks, recent_data)
        
        # ✅ Entrenar modelo surrogate periódicamente (cada 10 ciclos)
        if len(self.xai_explainer.training_data) % 10 == 0 and len(self.xai_explainer.training_data) >= 50:
            self.xai_explainer.train_surrogate_model()
        
        # ✅ Generar explicaciones SHAP
        shap_explanations = self.xai_explainer.explain_current_risk(var_risks)
        
        # Diagnóstico: Top influencers con SHAP values (CORREGIDO: zip seguro)
        if not shap_explanations:
            # Fallback: usar ranking simple por riesgo si no hay explicaciones SHAP
            top_influencers = [(var, risk, f"Riesgo: {risk:.1%}") for var, risk in 
                              sorted(var_risks.items(), key=lambda x: x[1], reverse=True)[:5]]
        else:
            # Combinar SHAP con riesgos de forma segura
            top_influencers = []
            shap_dict = {var: expl for var, _, expl in shap_explanations}  # Dict para lookup rápido
            for var, risk in sorted(var_risks.items(), key=lambda x: x[1], reverse=True)[:5]:
                expl = shap_dict.get(var, f"Riesgo: {risk:.1%}")
                top_influencers.append((var, risk, expl))
        
        # Salud global del sistema
        system_prob = np.mean([v for v in var_risks.values()]) if var_risks else 0.0
        
        status = 'normal'
        if system_prob >= self.config.ALERT_THRESHOLDS['critical']: status = 'critical'
        elif system_prob >= self.config.ALERT_THRESHOLDS['warning']: status = 'warning'
        
        # ✅ Calcular TTF global (mínimo TTF de variables críticas)
        critical_ttfs = [ttf['ttf_hours'] for ttf in var_ttf.values() if ttf['ttf_hours'] is not None and ttf['ttf_hours'] > 0]
        global_ttf = min(critical_ttfs) if critical_ttfs else None
        
        self.predictions = {
            'timestamp': datetime.now().isoformat(),
            'system_health': {
                'probability': float(system_prob),
                'status': status,
                'ttf_hours': global_ttf  # ✅ TTF global
            },
            'variable_risks': var_risks,
            'variable_ttf': var_ttf,  # ✅ TTF por variable
            'top_influencers': top_influencers,
            'shap_explanations': shap_explanations[:10],  # ✅ Top 10 explicaciones SHAP
            'alerts': self._generate_alerts(var_risks, var_ttf)
        }
        
        return self.predictions

    def generate_forecast(self, forecast_horizon_hours: int = 24) -> Dict[str, Any]:
        """
        ✅ MEJORA V11: Genera forecast de próximas N horas usando modelos SARIMAX/Prophet.
        
        Args:
            forecast_horizon_hours: Horizonte de forecast en horas (default: 24)
            
        Returns:
            Dict con:
            - forecast_data: DataFrame con valores proyectados por variable
            - forecast_probabilities: Probabilidades proyectadas por variable
            - forecast_timestamps: Timestamps futuros
            - system_forecast_prob: Probabilidad del sistema proyectada
        """
        self.logger.info(f"Generando forecast de próximas {forecast_horizon_hours} horas...")
        
        # Obtener último timestamp de los datos históricos
        last_timestamp = None
        dt_hours = 1.0
        
        # Intentar obtener desde los datos históricos más recientes
        if hasattr(self.baseline, 'results') and 'adaptive_limits' in self.baseline.results:
            # Buscar en los modelos si tienen time_meta
            for var_name in self.baseline.results['adaptive_limits'].keys():
                if var_name in self.baseline.models:
                    models = self.baseline.models[var_name]
                    if isinstance(models, dict) and 'time_meta' in models:
                        time_meta = models['time_meta']
                        last_timestamp = pd.to_datetime(time_meta.get('last_timestamp')) if time_meta.get('last_timestamp') else None
                        dt_hours = float(time_meta.get('dt_hours', 1.0))
                        if last_timestamp:
                            break
                    
                    # Fallback: inferir desde Prophet history
                    if 'prophet' in models and last_timestamp is None:
                        try:
                            hist = models['prophet'].history
                            if hist is not None and 'ds' in hist.columns and len(hist['ds']) >= 3:
                                ds_hist = pd.to_datetime(hist['ds']).sort_values()
                                last_timestamp = pd.to_datetime(ds_hist.iloc[-1])
                                diffs = ds_hist.diff().dropna().dt.total_seconds()
                                diffs = diffs[diffs > 0]
                                if not diffs.empty:
                                    dt_hours = float(diffs.median() / 3600.0)
                                    break
                        except Exception:
                            pass
        
        # Si no se pudo obtener, usar timestamp actual
        if last_timestamp is None:
            last_timestamp = pd.Timestamp.now()
            self.logger.warning("No se pudo obtener último timestamp histórico. Usando timestamp actual.")
        
        # Generar timestamps futuros
        forecast_timestamps = pd.date_range(
            start=last_timestamp + pd.Timedelta(hours=dt_hours),
            periods=forecast_horizon_hours,
            freq=pd.Timedelta(hours=dt_hours)
        )
        
        forecast_data = {}
        forecast_probabilities = {}
        
        # Generar forecast para cada variable con baseline
        for var_name in self.baseline.results.get('adaptive_limits', {}).keys():
            if var_name not in self.baseline.models:
                continue
            
            models = self.baseline.models[var_name]
            forecast_values = []
            
            # Intentar SARIMAX primero
            if 'sarima' in models:
                try:
                    sarima_model = models['sarima']
                    forecast = sarima_model.forecast(steps=forecast_horizon_hours)
                    forecast_values = forecast.values if hasattr(forecast, 'values') else list(forecast)
                    # Convertir a lista si es array de NumPy
                    if hasattr(forecast_values, 'tolist'):
                        forecast_values = forecast_values.tolist()
                    elif not isinstance(forecast_values, list):
                        forecast_values = list(forecast_values)
                except Exception as e:
                    self.logger.warning(f"SARIMAX forecast falló para {var_name}: {e}")
                    forecast_values = []
            
            # Fallback a Prophet
            # ✅ CORRECCIÓN: Verificar si forecast_values está vacío correctamente (no usar 'not' con arrays)
            if (not forecast_values or len(forecast_values) == 0) and 'prophet' in models:
                try:
                    future_df = pd.DataFrame({'ds': forecast_timestamps})
                    forecast = models['prophet'].predict(future_df)
                    forecast_values = forecast['yhat'].values.tolist()
                except Exception as e:
                    self.logger.warning(f"Prophet forecast falló para {var_name}: {e}")
            
            # Si no hay forecast, usar baseline
            # ✅ CORRECCIÓN: Verificar correctamente si está vacío (no usar 'not' con arrays)
            if not forecast_values or len(forecast_values) == 0:
                baseline_val = self.baseline.results['adaptive_limits'][var_name].get('baseline', 0.0)
                forecast_values = [baseline_val] * forecast_horizon_hours
            
            # Asegurar que forecast_values es una lista
            if not isinstance(forecast_values, list):
                if hasattr(forecast_values, 'tolist'):
                    forecast_values = forecast_values.tolist()
                else:
                    forecast_values = list(forecast_values)
            
            forecast_data[var_name] = forecast_values
            
            # Calcular probabilidad proyectada sobre el forecast
            # Usar la misma fórmula que el análisis histórico
            limits = self.baseline.results['adaptive_limits'][var_name]
            baseline_mean = limits.get('baseline', 0.0)
            range_size = limits.get('upper', baseline_mean) - limits.get('lower', baseline_mean)
            
            if range_size > 0 and len(forecast_values) > 0:
                # Calcular desviación media del forecast
                normalized_dev = [(v - baseline_mean) / range_size for v in forecast_values]
                mean_deviation = float(np.mean([abs(d) for d in normalized_dev]))
                deviation_factor = min(1.0, mean_deviation)
                
                # Calcular proporción fuera de límites en el forecast
                out_of_limits = float(np.mean([
                    (v > limits.get('upper', baseline_mean)) or (v < limits.get('lower', baseline_mean))
                    for v in forecast_values
                ]))
                limit_factor = min(1.0, out_of_limits)
                
                # Calcular tendencia del forecast
                if len(forecast_values) >= 2:
                    try:
                        x = np.arange(len(forecast_values))
                        slope, _ = np.polyfit(x, forecast_values, 1)
                        std_val = float(np.std(forecast_values))
                        if std_val > 1e-12:
                            trend_strength = abs(float(slope)) * len(forecast_values) / std_val
                            trend_factor = min(1.0, trend_strength)
                        else:
                            trend_factor = 0.0
                    except Exception:
                        trend_factor = 0.0
                else:
                    trend_factor = 0.0
                
                # Fórmula: 0.4*dev + 0.4*limit + 0.2*trend
                forecast_prob = (
                    0.4 * deviation_factor +
                    0.4 * limit_factor +
                    0.2 * trend_factor
                )
                forecast_probabilities[var_name] = float(forecast_prob)
            else:
                forecast_probabilities[var_name] = 0.0
        
        # Probabilidad del sistema proyectada
        system_forecast_prob = np.mean(list(forecast_probabilities.values())) if forecast_probabilities else 0.0
        
        return {
            'forecast_data': forecast_data,  # Dict[variable] -> List[valores]
            'forecast_probabilities': forecast_probabilities,  # Dict[variable] -> probabilidad
            'forecast_timestamps': forecast_timestamps.tolist(),  # List[timestamps]
            'system_forecast_prob': float(system_forecast_prob),
            'forecast_horizon_hours': forecast_horizon_hours
        }

    def _generate_alerts(self, var_risks: Dict[str, float], var_ttf: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        ✅ FASE 2 P1: Genera alertas enriquecidas con TTF.
        """
        alerts = []
        for var, prob in var_risks.items():
            if prob >= self.config.ALERT_THRESHOLDS['warning']:
                level = 'CRITICAL' if prob >= self.config.ALERT_THRESHOLDS['critical'] else 'WARNING'
                
                # ✅ Incluir TTF en el mensaje de alerta
                message = f"Riesgo elevado en {var} ({prob:.1%})"
                if var in var_ttf and var_ttf[var]['ttf_hours'] is not None:
                    ttf_hours = var_ttf[var]['ttf_hours']
                    if ttf_hours < 24:
                        message += f" - TTF: {ttf_hours:.1f}h ⚠️"
                    elif ttf_hours < 72:
                        message += f" - TTF: {ttf_hours/24:.1f} días"
                    else:
                        message += f" - TTF: {ttf_hours/24:.0f} días"
                
                alerts.append({
                    'variable': var,
                    'level': level,
                    'probability': prob,
                    'message': message,
                    'ttf_info': var_ttf.get(var)  # ✅ Información completa de TTF
                })
        return alerts
