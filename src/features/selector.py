import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any
from src.config import SystemConfig
from src.data.db_manager import DatabaseManager

class KeyVariableSelector:
    """Selector de variables con Rigor Extremo: Alineado con Investigación 2025."""
    
    def __init__(self, config: SystemConfig, db: DatabaseManager):
        self.config = config
        self.db = db
        self.results = {
            'selected_variables': {},
            'metrics': {},
            'last_scores': {} # Memoria de corto plazo para suavizado
        }
        self._setup_logging()

    def _setup_logging(self):
        self.logger = logging.getLogger(__name__)

    def calculate_variance_score(self, series: pd.Series) -> float:
        """Calcula el score de varianza logarítmica (Investigación Fase 2)."""
        try:
            unique_ratio = series.nunique() / len(series)
            if unique_ratio < 0.001: return 0.0
            variance = np.var(series)
            return float(np.log1p(abs(variance)) * unique_ratio)
        except Exception:
            return 0.0

    def _infer_dt_hours(self, raw_data: pd.DataFrame) -> float:
        """
        Inferir delta-t (en horas) desde columna temporal real.
        ✅ Requisito de agnosticismo: nunca asumir granularidad; usar timestamps.
        """
        try:
            date_col = next((c for c in ['timestamp', 'fecha', 'datetime', 'date'] if c in raw_data.columns), None)
            if not date_col:
                return 1.0
            ts = pd.to_datetime(raw_data[date_col], errors='coerce').dropna().sort_values()
            if len(ts) < 3:
                return 1.0
            diffs = ts.diff().dropna().dt.total_seconds()
            diffs = diffs[diffs > 0]
            if diffs.empty:
                return 1.0
            dt_hours = float(diffs.median() / 3600.0)
            return dt_hours if dt_hours > 0 else 1.0
        except Exception:
            return 1.0

    def select_critical_variables(self, raw_data: pd.DataFrame) -> List[str]:
        """
        ✅ CORRECCIÓN P0: Selección con detección dinámica de ventana temporal.
        Cambios: Rolling window adaptativo (no hardcodeado en 24).
        """
        self.logger.info("Iniciando selección de variables con SIMETRÍA MATEMÁTICA...")
        dt_hours = self._infer_dt_hours(raw_data)
        self.logger.info(f"✅ Delta-t inferido desde timestamp: {dt_hours:.6f} horas")
        df = raw_data.select_dtypes(include=[np.number])
        scores = {}
        
        # Eliminar timestamp si está presente para el análisis de correlación
        if 'timestamp' in df.columns: df = df.drop(columns=['timestamp'])
        
        # ✅ Detectar ventana óptima dinámicamente (basado en autocorrelación)
        optimal_window = self._detect_optimal_window(df, dt_hours=dt_hours)
        self.logger.info(f"✅ Ventana adaptativa detectada: {optimal_window} muestras (~{optimal_window*dt_hours:.2f}h)")

        for col in df.columns:
            # 1. Varianza (0.3)
            var_score = self.calculate_variance_score(df[col])
            if var_score == 0: continue
            
            # 2. ✅ Estabilidad (0.3) - Usa ventana ADAPTATIVA
            rolling_std = df[col].rolling(window=optimal_window, min_periods=1).std().mean()
            total_std = df[col].std()
            stability_score = 1 - (rolling_std / total_std) if total_std > 0 else 0
            
            # 3. Tendencia (0.2) - Pendiente por unidad de tiempo real (timestamp-driven)
            x = np.arange(len(df[col]), dtype=float) * float(dt_hours)
            slope = np.polyfit(x, df[col], 1)[0]
            trend_score = abs(slope)
            
            # 4. Correlación (0.2) - Promedio de correlación absoluta con el resto
            corr_score = df.corrwith(df[col]).abs().mean()
            
            # Puntuación final ponderada (Simetría V3)
            current_score = (
                0.3 * var_score + 
                0.3 * stability_score + 
                0.2 * trend_score + 
                0.2 * corr_score
            )
            
            # Suavizado de Score (PLC Rigor): Mezclar con score previo (70/30)
            prev_score = self.results['last_scores'].get(col, current_score)
            blended_score = (prev_score * 0.7) + (current_score * 0.3)
            self.results['last_scores'][col] = blended_score
            
            scores[col] = {
                'variance': float(var_score),
                'stability': float(stability_score),
                'trend': float(trend_score),
                'correlation': float(corr_score),
                'final_score': float(blended_score)
            }

        if not scores:
            self.logger.warning("No se identificaron variables válidas.")
            return []

        # Categorización Bucketing (80th/50th Percentile)
        all_final_scores = pd.Series({k: v['final_score'] for k, v in scores.items()})
        threshold_critical = all_final_scores.quantile(0.8)
        threshold_monitoring = all_final_scores.quantile(0.5)

        critical_vars = []
        for var, details in scores.items():
            if details['final_score'] >= threshold_critical:
                details['category'] = 'critical'
                critical_vars.append(var)
            elif details['final_score'] >= threshold_monitoring:
                details['category'] = 'monitoring'
            else:
                details['category'] = 'discarded'

        self.results['variables'] = scores
        
        # PERSISTENCIA PARA AUDITORÍA DE DIAMANTE
        self.db.save_selection_metrics(scores)
        
        self.logger.info(f"Seleccionadas {len(critical_vars)} variables CRÍTICAS (Investigación 2025).")
        return critical_vars
    
    def get_variable_categories(self) -> Dict[str, str]:
        """
        ✅ MEJORA V14: Retorna categorías completas de todas las variables.
        Útil para detectar cambios en categorías (descartada → crítica, etc.)
        """
        if 'variables' not in self.results:
            return {}
        
        categories = {}
        for var, details in self.results['variables'].items():
            categories[var] = details.get('category', 'discarded')
        
        return categories
    
    def _detect_optimal_window(self, df: pd.DataFrame, dt_hours: float = 1.0) -> int:
        """
        ✅ CORRECCIÓN P0: Detecta ventana óptima para rolling statistics.
        Estrategia: Buscar primer mínimo en autocorrelación promedio.
        """
        # Probar ventanas de 6 a 168 horas (1/4 día hasta 7 días), convertidas a muestras usando dt_hours real
        max_samples = min(len(df), 500)  # Limitar para velocidad
        sample_cols = df.sample(n=min(5, len(df.columns)), axis=1, random_state=42)
        
        # Calcular autocorrelación promedio para diferentes ventanas
        candidate_hours = [6, 12, 24, 48, 72, 96, 120, 168]
        candidate_windows = sorted({max(3, int(round(h / max(dt_hours, 1e-9)))) for h in candidate_hours})
        best_window = candidate_windows[0] if candidate_windows else 24
        min_autocorr = float('inf')
        
        for window in candidate_windows:
            if window > len(df) // 4:  # No usar ventanas mayores al 25% de los datos
                break
                
            autocorr_values = []
            for col in sample_cols.columns:
                try:
                    rolling_mean = sample_cols[col].rolling(window=window).mean()
                    # Correlación entre valores y su rolling mean
                    corr = sample_cols[col].corr(rolling_mean)
                    if not np.isnan(corr):
                        autocorr_values.append(abs(1 - corr))  # Queremos decorrelación
                except Exception:
                    continue
            
            if autocorr_values:
                avg_decorr = np.mean(autocorr_values)
                if avg_decorr < min_autocorr:
                    min_autocorr = avg_decorr
                    best_window = window
        
        return best_window