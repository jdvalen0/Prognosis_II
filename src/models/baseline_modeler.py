import logging
import warnings
import pandas as pd
import numpy as np
import json
import itertools
from sqlalchemy import text
from statsmodels.tsa.stattools import acf, adfuller, kpss
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from sklearn.ensemble import IsolationForest
from typing import Dict, Any, Optional, Tuple, List
from src.config import SystemConfig
from src.data.db_manager import DatabaseManager

class BaselineModeler:
    """Modelador inteligente: optimiza el ensamble y detecta anomalías sin sesgo."""
    
    def __init__(self, config: SystemConfig, db: DatabaseManager):
        self.config = config
        self.db = db
        self.models = {}  # REQUERIDO PARA INFERENCIA
        self.results = {'adaptive_limits': {}}
        self._setup_logging()

    def _setup_logging(self):
        self.logger = logging.getLogger(__name__)
    
    def has_baseline(self, variable_name: str) -> bool:
        """
        ✅ MEJORA V14: Verifica si una variable tiene baseline entrenado.
        Retorna True si existe baseline en BD o en memoria.
        """
        # Verificar en memoria primero (más rápido)
        if variable_name in self.results.get('adaptive_limits', {}):
            return True
        
        # Verificar en BD
        baseline_stats = self.db.get_baseline_stats(variable_name)
        return len(baseline_stats) > 0
    
    def get_variables_with_baseline(self) -> List[str]:
        """
        ✅ MEJORA V14: Retorna lista de variables que tienen baseline entrenado.
        Útil para comparar con nuevas variables críticas y detectar cambios.
        """
        variables_with_baseline = []
        
        # Variables en memoria
        variables_with_baseline.extend(list(self.results.get('adaptive_limits', {}).keys()))
        
        # Variables en BD (verificar todas)
        try:
            with self.db.engine.connect() as conn:
                result = conn.execute(
                    text("SELECT variable_name FROM baseline_models")
                ).fetchall()
                for row in result:
                    var_name = row[0]
                    if var_name not in variables_with_baseline:
                        variables_with_baseline.append(var_name)
        except Exception as e:
            self.logger.warning(f"Error obteniendo variables con baseline desde BD: {e}")
        
        return list(set(variables_with_baseline))  # Eliminar duplicados

    def _infer_dt_hours_from_index(self, series: pd.Series) -> Optional[float]:
        """
        Inferir delta-t (en horas) desde el índice temporal real.
        ✅ Agnosticismo total: usar timestamps reales, no asumir horas.
        """
        try:
            if not isinstance(series.index, pd.DatetimeIndex):
                return None
            idx = series.index.dropna().sort_values()
            if len(idx) < 3:
                return None
            diffs = idx.to_series().diff().dropna().dt.total_seconds()
            diffs = diffs[diffs > 0]
            if diffs.empty:
                return None
            dt_hours = float(diffs.median() / 3600.0)
            return dt_hours if dt_hours > 0 else None
        except Exception:
            return None

    def _infer_freq_string(self, series: pd.Series) -> Optional[str]:
        """
        Infiere la frecuencia de pandas desde el índice temporal.
        Retorna un string de frecuencia válido para statsmodels (e.g., 'H', 'min', 'D').
        """
        try:
            if not isinstance(series.index, pd.DatetimeIndex):
                return None
            
            # Intentar inferir frecuencia usando pandas
            inferred_freq = pd.infer_freq(series.index)
            if inferred_freq:
                return inferred_freq
            
            # Si pandas no puede inferir, calcular desde dt_hours
            dt_hours = self._infer_dt_hours_from_index(series)
            if dt_hours is None:
                return None
            
            # Mapear dt_hours a frecuencia de pandas
            if dt_hours < 0.017:  # < 1 minuto
                return 'min'
            elif dt_hours < 1.0:  # < 1 hora
                return 'min'
            elif dt_hours < 24.0:  # < 1 día
                return 'H'
            else:  # >= 1 día
                return 'D'
        except Exception:
            return None

    def _prepare_series_for_sarimax(self, series: pd.Series) -> pd.Series:
        """
        Prepara la serie para SARIMAX asegurando que tenga frecuencia definida.
        Esto elimina el warning "No frequency information was provided".
        """
        if not isinstance(series.index, pd.DatetimeIndex):
            # Si no es DatetimeIndex, intentar convertir
            try:
                series.index = pd.to_datetime(series.index)
            except Exception:
                # Si no se puede convertir, crear un índice temporal simulado
                # (esto no debería pasar en producción, pero es un fallback seguro)
                self.logger.warning(f"Serie sin DatetimeIndex, creando índice simulado para {series.name}")
                series.index = pd.date_range(start='2020-01-01', periods=len(series), freq='H')
        
        # Asegurar que la frecuencia esté definida
        if series.index.freq is None:
            freq_str = self._infer_freq_string(series)
            if freq_str:
                try:
                    series = series.asfreq(freq_str, method='ffill')
                except Exception:
                    # Si asfreq falla, al menos establecer freq en el índice
                    series.index.freq = pd.tseries.frequencies.to_offset(freq_str)
            else:
                # Fallback: usar frecuencia horaria si no se puede inferir
                self.logger.warning(f"No se pudo inferir frecuencia para {series.name}, usando 'H' como fallback")
                series.index.freq = pd.tseries.frequencies.to_offset('H')
        
        return series

    def _check_stationarity(self, series: pd.Series) -> bool:
        """Realiza pruebas ADF y KPSS para determinar estacionariedad (Fase 3)."""
        try:
            # ADF (Ho: La serie tiene raíz unitaria - no estacionaria)
            adf_res = adfuller(series.dropna())
            is_adf_stationary = adf_res[1] < self.config.STATISTICAL_THRESHOLDS['stationarity_pvalue']
            
            # KPSS (Ho: La serie es estacionaria)
            # Suprimir warnings de interpolación que son esperados cuando la serie es muy no estacionaria
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', message='.*InterpolationWarning.*')
                warnings.filterwarnings('ignore', message='.*outside of the range of p-values.*')
                kpss_res = kpss(series.dropna())
            is_kpss_stationary = kpss_res[1] > self.config.STATISTICAL_THRESHOLDS['stationarity_pvalue']
            
            self.logger.info(f"Test Estacionariedad: ADF={is_adf_stationary}, KPSS={is_kpss_stationary}")
            # Consideramos estacionaria si pasa alguna para ser flexibles pero rigurosos
            return is_adf_stationary or is_kpss_stationary
        except Exception as e:
            self.logger.error(f"Error en tests de estacionariedad: {e}")
            return False

    def _check_seasonality(self, series: pd.Series) -> Tuple[bool, int]:
        """
        ✅ CORRECCIÓN P0: Detecta estacionalidad DINÁMICA usando ACF + periodogram.
        Returns: (tiene_seasonality, periodo_detectado)
        """
        # Si tenemos índice temporal, no podemos usar "24" como muestras; usamos días en tiempo real.
        dt_hours = self._infer_dt_hours_from_index(series)
        if dt_hours is None:
            dt_hours = 1.0  # fallback seguro si no hay timestamp; se espera timestamp en producción

        min_samples = max(12, int(round(24.0 / max(dt_hours, 1e-9))))  # mínimo ~24h o 12 muestras
        if len(series) < min_samples:
            return False, 0
        
        series_clean = series.dropna()
        if len(series_clean) < 12:  # Mínimo para ACF
            return False, 0
        
        # 1. ACF: Detectar picos de autocorrelación
        # Usar hasta 7 días en unidades de muestras: 7*24h / dt_hours
        max_lags_time = int(round((7 * 24.0) / max(dt_hours, 1e-9)))
        max_lags = min(len(series_clean)//2, max(2, max_lags_time))
        if max_lags < 2:
            return False, 0
            
        try:
            auto_corr = acf(series_clean, nlags=max_lags, fft=True)
            if len(auto_corr) < 2:
                return False, 0
        except Exception as e:
            self.logger.warning(f"Error calculando ACF: {e}")
            return False, 0
        
        # 2. Buscar el primer pico significativo (excluyendo lag=0)
        significant_lags = np.where(abs(auto_corr[1:]) > 0.3)[0] + 1  # Umbral más robusto
        
        if len(significant_lags) == 0:
            return False, 0
        
        # 3. Identificar el periodo dominante (primer pico fuerte)
        detected_period = int(significant_lags[0])
        
        # 4. Validación: Si el pico es muy bajo, verificar con periodogram
        if abs(auto_corr[detected_period]) < 0.5:
            try:
                from scipy.signal import periodogram
                freqs, power = periodogram(series.dropna().values)
                # Excluir frecuencia 0 (tendencia DC)
                dominant_freq_idx = np.argmax(power[1:]) + 1
                if power[dominant_freq_idx] > np.mean(power) * 3:  # 3x más potencia que la media
                    # Convertir frecuencia a periodo
                    detected_period = int(1 / freqs[dominant_freq_idx]) if freqs[dominant_freq_idx] > 0 else detected_period
            except Exception as e:
                self.logger.warning(f"Periodogram falló: {e}")
        
        # 5. Validación de rangos razonables (entre ~2h y 7 días en tiempo real)
        detected_hours = float(detected_period) * float(dt_hours)
        if 2.0 <= detected_hours <= (7 * 24.0):
            self.logger.info(
                f"✅ Seasonality detectada: periodo={detected_period} muestras (~{detected_hours:.2f}h, ACF={auto_corr[detected_period]:.3f})"
            )
            return True, detected_period
        
        return False, 0
    
    def _optimize_sarima_params(self, data: pd.Series, seasonal_period: int) -> Tuple[tuple, tuple, float]:
        """
        ✅ CORRECCIÓN P0: Auto-optimización de parámetros SARIMAX usando AIC.
        Estrategia: Grid Search limitado para velocidad industrial.
        Returns: (best_order, best_seasonal_order, best_aic)
        """
        # Preparar serie con frecuencia definida para evitar warnings
        data = self._prepare_series_for_sarimax(data)
        
        # Grid reducido para velocidad (3^3 * 2^3 = 216 combinaciones máx)
        p_range = [0, 1, 2]
        d_range = [0, 1]
        q_range = [0, 1, 2]
        
        # Para seasonal: solo probar (P,D,Q) básicos
        P_range = [0, 1]
        D_range = [1]  # Diferenciación estacional casi siempre necesaria
        Q_range = [0, 1]
        
        best_aic = np.inf
        best_order = (1, 1, 1)
        best_seasonal = (1, 1, 1, seasonal_period) if seasonal_period > 0 else (0, 0, 0, 0)
        
        # Limitar búsqueda si datos son pocos (evitar overfitting)
        if len(data) < 200:
            p_range = [0, 1]
            q_range = [0, 1]
            P_range = [0, 1]
            Q_range = [0, 1]
        
        tested_count = 0
        max_tests = 50  # Límite de tiempo para producción
        
        for p, d, q in itertools.product(p_range, d_range, q_range):
            if tested_count >= max_tests:
                break
                
            for P, D, Q in itertools.product(P_range, D_range, Q_range):
                if tested_count >= max_tests:
                    break
                    
                try:
                    # Suprimir warnings de convergencia y frecuencia durante la optimización
                    with warnings.catch_warnings():
                        warnings.filterwarnings('ignore', message='.*No frequency information.*')
                        warnings.filterwarnings('ignore', message='.*ConvergenceWarning.*')
                        warnings.filterwarnings('ignore', category=UserWarning)
                        
                        model = SARIMAX(
                            data,
                            order=(p, d, q),
                            seasonal_order=(P, D, Q, seasonal_period) if seasonal_period > 0 else (0, 0, 0, 0),
                            enforce_stationarity=False,
                            enforce_invertibility=False
                        )
                        fitted = model.fit(disp=False, maxiter=50, method='lbfgs')
                    
                    if fitted.aic < best_aic:
                        best_aic = fitted.aic
                        best_order = (p, d, q)
                        best_seasonal = (P, D, Q, seasonal_period) if seasonal_period > 0 else (0, 0, 0, 0)
                    
                    tested_count += 1
                    
                except Exception:
                    continue
        
        if best_aic == np.inf:
            self.logger.warning(f"No se pudo optimizar SARIMAX (todas las combinaciones fallaron). Usando parámetros por defecto.")
            best_aic = None

        if best_aic is not None:
            self.logger.info(f"✅ SARIMAX optimizado: order={best_order}, seasonal={best_seasonal}, AIC={best_aic:.2f} ({tested_count} pruebas)")
        else:
            self.logger.info(f"⚠️ SARIMAX usando parámetros por defecto: order={best_order}, seasonal={best_seasonal} ({tested_count} pruebas fallidas)")
        return best_order, best_seasonal, best_aic

    def fit_ensemble(self, variable_name: str, data: pd.Series) -> Dict[str, Any]:
        """
        ✅ CORRECCIÓN P0: Ensamble con auto-optimización y seasonality dinámica.
        Cambios respecto a versión anterior:
        - SARIMAX: parámetros optimizados por AIC (no hardcodeados)
        - Seasonality: periodo detectado dinámicamente (no fijo en 24)
        - Límites: basados en percentiles robustos (no ±3σ)
        """
        self.logger.info(f"Entrenando ensamble RIGUROSO para {variable_name}...")
        models = {}
        
        if len(data) < self.config.MODEL_PARAMS.get('min_data_points', 100):
            self.logger.warning(f"Batch demasiado pequeño ({len(data)} pts) para entrenamiento de modelos. Usando estadísticas básicas.")
            # Usar percentiles en lugar de ±3σ
            return {
                'variable': variable_name,
                'models': {},
                'limits': {
                    'baseline': float(data.median()),  # Mediana más robusta que media
                    'upper': float(data.quantile(0.95)),  # P95 en lugar de mean+3σ
                    'lower': float(data.quantile(0.05))   # P5 en lugar de mean-3σ
                },
                'has_complex_patterns': False,
                'is_stationary': True,
                'seasonal_period': 0
            }

        # ✅ Asegurar orden temporal si viene con índice datetime
        if isinstance(data.index, pd.DatetimeIndex):
            data = data.sort_index()

        is_stationary = self._check_stationarity(data)
        has_patterns, seasonal_period = self._check_seasonality(data)
        
        # 1. SARIMAX con optimización automática
        if has_patterns or not is_stationary:
            try:
                # Preparar serie con frecuencia definida
                data_prepared = self._prepare_series_for_sarimax(data.copy())
                best_order, best_seasonal, best_aic = self._optimize_sarima_params(data_prepared, seasonal_period)
                
                # Suprimir warnings durante el entrenamiento final
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', message='.*No frequency information.*')
                    warnings.filterwarnings('ignore', message='.*ConvergenceWarning.*')
                    warnings.filterwarnings('ignore', category=UserWarning)
                    
                    sarima = SARIMAX(
                        data_prepared, 
                        order=best_order, 
                        seasonal_order=best_seasonal,
                        enforce_stationarity=False, 
                        enforce_invertibility=False
                    )
                    fitted_sarima = sarima.fit(disp=False, low_memory=True, maxiter=100)
                models['sarima'] = fitted_sarima
                
                # Métricas de performance
                models['sarima_metrics'] = {
                    'aic': float(fitted_sarima.aic),
                    'bic': float(fitted_sarima.bic),
                    'order': best_order,
                    'seasonal_order': best_seasonal
                }
                
                self.logger.info(f"✅ SARIMAX optimizado para {variable_name}: AIC={best_aic:.2f}")
            except Exception as e:
                self.logger.warning(f"SARIMAX falló para {variable_name}: {e}. Usando Prophet como alternativa.")

        # 2. Prophet con configuración adaptativa
        if has_patterns or len(data) > 200:
            try:
                # ✅ Requisito: usar índice temporal real; no simular timestamps
                if not isinstance(data.index, pd.DatetimeIndex):
                    raise ValueError(
                        "Prophet requiere índice temporal real (DatetimeIndex). "
                        "Asegura que el pipeline entregue timestamp como índice."
                    )
                ds = data.index

                prophet_df = pd.DataFrame({'ds': ds, 'y': data.values})
                
                dt_hours = self._infer_dt_hours_from_index(data) or 1.0
                seasonal_hours = float(seasonal_period) * float(dt_hours) if seasonal_period else 0.0
                
                # Evitar división por cero al calcular seasonality_mode
                data_mean = data.mean()
                if abs(data_mean) < 1e-10:
                    seasonality_mode = 'additive'  # Fallback seguro si media es ~0
                else:
                    cv = data.std() / data_mean  # Coeficiente de variación
                    seasonality_mode = 'multiplicative' if cv > 0.3 else 'additive'
                
                m = Prophet(
                    daily_seasonality=(20.0 <= seasonal_hours <= 28.0),
                    weekly_seasonality=(140.0 <= seasonal_hours <= 196.0),
                    yearly_seasonality=False,
                    seasonality_mode=seasonality_mode
                )
                m.fit(prophet_df)
                models['prophet'] = m

                # Persistir metadatos temporales para forecasting sin asumir "now"
                models['time_meta'] = {
                    'last_timestamp': str(ds[-1]),
                    'dt_hours': float(dt_hours)
                }
                
                self.logger.info(f"✅ Prophet ajustado para {variable_name}")
            except Exception as e:
                self.logger.warning(f"Prophet falló para {variable_name}: {e}")

        # 3. Isolation Forest (detección de anomalías contextuales)
        iso = IsolationForest(
            contamination=self.config.STATISTICAL_THRESHOLDS['anomaly_contamination'],
            random_state=42,
            n_estimators=100
        )
        iso.fit(data.values.reshape(-1, 1))
        models['isolation_forest'] = iso
        
        # 4. ✅ CORRECCIÓN P0: Límites adaptativos basados en PERCENTILES (no ±3σ)
        limits = {
            'baseline': float(data.median()),      # Mediana: más robusta a outliers
            'upper': float(data.quantile(0.95)),   # Percentil 95 (no mean+3σ)
            'lower': float(data.quantile(0.05)),   # Percentil 5 (no mean-3σ)
            'p99': float(data.quantile(0.99)),     # Para alertas críticas
            'p01': float(data.quantile(0.01))      # Para alertas críticas
        }
        
        return {
            'variable': variable_name,
            'models': models,
            'limits': limits,
            'has_complex_patterns': has_patterns,
            'is_stationary': is_stationary,
            'seasonal_period': seasonal_period
        }

    def get_forecast(self, variable_name: str, steps: int = 1) -> float:
        """
        ✅ CORRECCIÓN P0: Genera proyección usando SARIMAX/Prophet (no solo baseline.mean()).
        Estrategia de Ensamble Jerárquico:
        1. SARIMAX (si disponible y converge)
        2. Prophet (fallback robusto para tendencias/seasonality)
        3. Baseline media (último recurso si no hay modelos)
        """
        if variable_name not in self.models:
            return self.results['adaptive_limits'].get(variable_name, {}).get('baseline', 0.0)
            
        models = self.models[variable_name]
        
        # 1. PRIORIDAD: SARIMAX para series estacionarias/estacionales con patrones claros
        if 'sarima' in models:
            try:
                sarima_model = models['sarima']
                forecast = sarima_model.forecast(steps=steps)
                predicted_value = float(forecast.iloc[-1]) if hasattr(forecast, 'iloc') else float(forecast[-1])
                self.logger.debug(f"SARIMAX forecast para {variable_name}: {predicted_value:.4f}")
                return predicted_value
            except Exception as e:
                self.logger.warning(f"SARIMAX forecast falló para {variable_name}: {e}. Intentando Prophet...")
        
        # 2. FALLBACK: Prophet para robustez ante datos irregulares/missing values
        if 'prophet' in models:
            try:
                # ✅ Forecast en timeline real: usar último timestamp + dt inferido
                time_meta = models.get('time_meta', {}) if isinstance(models, dict) else {}
                last_ts = pd.to_datetime(time_meta.get('last_timestamp')) if time_meta.get('last_timestamp') else None
                dt_hours = float(time_meta.get('dt_hours', 0.0))

                # Fallback científico: inferir desde history del modelo Prophet (sin usar "now")
                if last_ts is None or dt_hours <= 0:
                    try:
                        hist = models['prophet'].history
                        if hist is not None and 'ds' in hist.columns and len(hist['ds']) >= 3:
                            ds_hist = pd.to_datetime(hist['ds']).sort_values()
                            last_ts = pd.to_datetime(ds_hist.iloc[-1])
                            diffs = ds_hist.diff().dropna().dt.total_seconds()
                            diffs = diffs[diffs > 0]
                            if not diffs.empty:
                                dt_hours = float(diffs.median() / 3600.0)
                    except Exception:
                        pass

                if last_ts is None or dt_hours <= 0:
                    raise ValueError("time_meta incompleto para Prophet; no se pudo inferir last_timestamp/dt_hours desde history.")
                future = pd.DataFrame({
                    'ds': [last_ts + pd.Timedelta(hours=dt_hours * (i + 1)) for i in range(steps)]
                })
                forecast = models['prophet'].predict(future)
                predicted_value = float(forecast['yhat'].iloc[-1])
                self.logger.debug(f"Prophet forecast para {variable_name}: {predicted_value:.4f}")
                return predicted_value
            except Exception as e:
                self.logger.warning(f"Prophet forecast falló para {variable_name}: {e}. Usando baseline media...")
                
        # 3. ÚLTIMO RECURSO: Baseline media (solo si no hay modelos disponibles/funcionales)
        baseline_value = self.results['adaptive_limits'].get(variable_name, {}).get('baseline', 0.0)
        self.logger.debug(f"Usando baseline media para {variable_name}: {baseline_value:.4f}")
        return baseline_value

    def get_anomaly_score(self, variable_name: str, value: float) -> float:
        """Calcula el score de anomalía contextual usando Isolation Forest."""
        if variable_name in self.models and 'isolation_forest' in self.models[variable_name]:
            iso = self.models[variable_name]['isolation_forest']
            # Isolation Forest score: más bajo es más anómalo
            score = iso.decision_function([[value]])[0]
            return float(score)
        return 0.0

    def update_incremental(self, variable_name: str, new_stats: Dict[str, float]):
        """
        ✅ CORRECCIÓN P0: Actualización incremental con estadísticas robustas.
        Alineación Incremental 70/30: Mezcla de pesos (Simetría Fase 3).
        Cambiado de mean/std a median/iqr para robustez.
        Incluye migración automática de formato antiguo.
        """
        current_model = self.db.get_baseline_stats(variable_name)
        if not current_model:
            return new_stats
        
        # ✅ MIGRACIÓN AUTOMÁTICA: Convertir formato antiguo (mean/std) a nuevo (median/iqr)
        if 'mean' in current_model and 'median' not in current_model:
            self.logger.info(f"Migrando formato antiguo a nuevo para {variable_name}")
            mean_val = current_model.get('mean', 0)
            std_val = current_model.get('std', 0)
            # Aproximación: para distribución normal, IQR ≈ 1.35*std, pero usamos 6*std como aproximación conservadora
            current_model = {
                'median': mean_val,
                'iqr': std_val * 6,  # Aproximación conservadora
                'p95': mean_val + 1.96 * std_val,  # Percentil 95 para normal
                'p05': mean_val - 1.96 * std_val   # Percentil 5 para normal
            }
            
        weight_old = 0.7
        weight_new = 0.3
        
        updated_stats = {}
        # Usar median/iqr en lugar de mean/std
        for key in ['median', 'iqr', 'p95', 'p05']:
            old_val = current_model.get(key, new_stats.get(key, 0))
            new_val = new_stats.get(key, old_val)
            updated_stats[key] = (old_val * weight_old) + (new_val * weight_new)
            
        return updated_stats

    def save_baseline(self, results: Dict[str, Any]):
        """
        ✅ CORRECCIÓN P0: Persiste metadatos extendidos (seasonal_period, métricas AIC/BIC).
        Cambios: Incluye periodo estacional detectado y métricas de optimización.
        """
        var = results['variable']
        
        # Usar mediana y rango intercuartil para estadísticas robustas
        new_stats = {
            'median': results['limits']['baseline'],
            'iqr': results['limits']['upper'] - results['limits']['lower'],
            'p95': results['limits'].get('p95', results['limits']['upper']),
            'p05': results['limits'].get('p05', results['limits']['lower'])
        }
        
        # Lógica Incremental 70/30
        final_stats = self.update_incremental(var, new_stats)
        
        if 'models' in results:
            self.models[var] = results['models']
            
        with self.db.engine.connect() as conn:
            with conn.begin():
                # 1. Asegurar Tablas de Rigor (con columnas extendidas)
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS pattern_registry (
                        variable_name VARCHAR PRIMARY KEY,
                        has_seasonality BOOLEAN,
                        seasonal_period INTEGER DEFAULT 0,
                        stationarity_status VARCHAR,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """))
                
                # 2. Insertar/Actualizar baseline con estadísticas robustas
                conn.execute(text("""
                    INSERT INTO baseline_models (variable_name, baseline_stats)
                    VALUES (:var, :stats)
                    ON CONFLICT (variable_name) DO UPDATE SET 
                        baseline_stats = EXCLUDED.baseline_stats,
                        updated_at = CURRENT_TIMESTAMP
                """), {'var': var, 'stats': json.dumps(final_stats)})
                
                # 3. Actualizar registro de patrones con periodo estacional
                conn.execute(text("""
                    INSERT INTO pattern_registry (variable_name, has_seasonality, seasonal_period, stationarity_status)
                    VALUES (:var, :pattern, :period, :stat)
                    ON CONFLICT (variable_name) DO UPDATE SET 
                        has_seasonality = EXCLUDED.has_seasonality,
                        seasonal_period = EXCLUDED.seasonal_period,
                        stationarity_status = EXCLUDED.stationarity_status,
                        updated_at = CURRENT_TIMESTAMP
                """), {
                    'var': var, 
                    'pattern': results['has_complex_patterns'],
                    'period': results.get('seasonal_period', 0),
                    'stat': 'stationary' if results.get('is_stationary') else 'non-stationary'
                })
        
        # Guardar Versión MLOps con parámetros optimizados
        sarima_metrics = results.get('models', {}).get('sarima_metrics', {})
        model_params = {
            'sarima_order': sarima_metrics.get('order', (1, 1, 1)),
            'sarima_seasonal': sarima_metrics.get('seasonal_order', (0, 0, 0, 0)),
            'seasonal_period': results.get('seasonal_period', 0),
            'prophet_enabled': 'prophet' in results.get('models', {})
        }
        performance_metrics = {
            'baseline_median': final_stats.get('median', 0),
            'iqr': final_stats.get('iqr', 0),
            'has_complex_patterns': results['has_complex_patterns'],
            'aic': sarima_metrics.get('aic'),
            'bic': sarima_metrics.get('bic')
        }
        self.db.save_model_version(var, model_params, performance_metrics)
        
        # Actualizar memoria interna para el predictor
        self.results['adaptive_limits'][var] = results['limits']
