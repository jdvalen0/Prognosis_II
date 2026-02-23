"""
✅ FASE 3 P2: Sistema de Monitoreo Continuo y Alertas
Monitoreo en tiempo real de performance, drift detection y alertas automatizadas.
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from sqlalchemy import text


class AlertSeverity(Enum):
    """Niveles de severidad de alertas"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class DriftType(Enum):
    """Tipos de drift detectables"""
    CONCEPT_DRIFT = "concept_drift"  # Cambio en P(Y|X)
    DATA_DRIFT = "data_drift"  # Cambio en P(X)
    PERFORMANCE_DRIFT = "performance_drift"  # Degradación de métricas


@dataclass
class Alert:
    """Alerta del sistema de monitoreo"""
    alert_id: str
    timestamp: str
    severity: str
    category: str  # "drift", "performance", "anomaly", "system"
    title: str
    message: str
    affected_variables: List[str]
    metrics: Dict[str, Any]
    recommended_action: str
    auto_resolved: bool = False


class ContinuousMonitor:
    """
    Sistema de monitoreo continuo para Prognosis II.
    Funcionalidades:
    - Detección de concept drift y data drift
    - Monitoreo de performance de modelos
    - Alertas automatizadas con severidad
    - Registro de eventos para auditoría
    - Recomendaciones de re-entrenamiento
    """
    
    def __init__(self, db_manager, config):
        self.db = db_manager
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.baseline_history = {}  # Variable → histórico de estadísticas
        self.performance_history = {}  # Variable → histórico de métricas
        self.alert_buffer = []
        self._ensure_monitoring_schema()
        
    def _ensure_monitoring_schema(self):
        """Crea esquema de monitoreo si no existe"""
        with self.db.engine.connect() as conn:
            with conn.begin():
                # Tabla de eventos de monitoreo
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS monitoring_events (
                        event_id SERIAL PRIMARY KEY,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        event_type VARCHAR NOT NULL,
                        severity VARCHAR,
                        affected_variables JSONB,
                        metrics JSONB,
                        details TEXT,
                        resolved BOOLEAN DEFAULT FALSE
                    );
                """))
                
                # Tabla de métricas de performance
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS monitoring_performance_metrics (
                        metric_id SERIAL PRIMARY KEY,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        variable_name VARCHAR,
                        model_type VARCHAR,
                        metric_name VARCHAR,
                        metric_value FLOAT,
                        baseline_value FLOAT
                    );
                """))
                
                # Tabla de drift detection
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS monitoring_drift_detection (
                        drift_id SERIAL PRIMARY KEY,
                        detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        variable_name VARCHAR,
                        drift_type VARCHAR,
                        drift_score FLOAT,
                        statistical_test VARCHAR,
                        p_value FLOAT,
                        is_significant BOOLEAN
                    );
                """))
        
        self.logger.info("✅ Monitoring schema inicializado")
    
    def detect_data_drift(self, 
                         variable_name: str, 
                         reference_data: pd.Series, 
                         current_data: pd.Series) -> Dict[str, Any]:
        """
        Detecta data drift usando test estadístico (Kolmogorov-Smirnov).
        Data drift = cambio en la distribución P(X).
        """
        from scipy.stats import ks_2samp, wasserstein_distance
        
        try:
            # Test de Kolmogorov-Smirnov
            ks_stat, ks_pvalue = ks_2samp(reference_data.dropna(), current_data.dropna())
            
            # Wasserstein distance (Earth Mover's Distance)
            wasserstein_dist = wasserstein_distance(reference_data.dropna(), current_data.dropna())
            
            # Criterio: p-value < 0.05 indica drift significativo
            is_drift = ks_pvalue < 0.05
            
            drift_info = {
                'variable': variable_name,
                'drift_type': DriftType.DATA_DRIFT.value,
                'ks_statistic': float(ks_stat),
                'ks_pvalue': float(ks_pvalue),
                'wasserstein_distance': float(wasserstein_dist),
                'is_significant': is_drift,
                'severity': AlertSeverity.CRITICAL.value if is_drift else AlertSeverity.INFO.value
            }
            
            # Registrar en DB
            self._log_drift_detection(drift_info)
            
            if is_drift:
                self.logger.warning(f"⚠️ Data drift detectado en {variable_name}: p={ks_pvalue:.4f}")
                self._create_alert(
                    severity=AlertSeverity.CRITICAL,
                    category="drift",
                    title=f"Data Drift Detectado: {variable_name}",
                    message=f"Distribución de {variable_name} ha cambiado significativamente (p={ks_pvalue:.4f})",
                    affected_variables=[variable_name],
                    metrics=drift_info,
                    recommended_action="Considerar re-entrenamiento del modelo con datos recientes"
                )
            
            return drift_info
            
        except Exception as e:
            self.logger.error(f"Error detectando data drift: {e}")
            return {'error': str(e)}
    
    def detect_concept_drift(self, 
                            variable_name: str, 
                            predictions: np.ndarray, 
                            actuals: np.ndarray,
                            window_size: int = 50) -> Dict[str, Any]:
        """
        Detecta concept drift usando Page-Hinkley test.
        Concept drift = cambio en P(Y|X) (relación input→output).
        """
        from sklearn.metrics import mean_absolute_error
        
        try:
            if len(predictions) < window_size:
                return {'status': 'insufficient_data'}
            
            # Calcular error residual
            errors = np.abs(predictions - actuals)
            
            # Page-Hinkley test: detecta cambios en la media
            threshold = 10  # Umbral de detección
            delta = 0.005  # Magnitud mínima de cambio
            
            cumsum = 0
            min_cumsum = 0
            drift_point = None
            
            for i, error in enumerate(errors):
                cumsum += error - np.mean(errors) - delta
                if cumsum < min_cumsum:
                    min_cumsum = cumsum
                
                diff = cumsum - min_cumsum
                if diff > threshold:
                    drift_point = i
                    break
            
            is_drift = drift_point is not None
            
            # Calcular MAE antes y después del drift point (si existe)
            if is_drift:
                mae_before = mean_absolute_error(actuals[:drift_point], predictions[:drift_point])
                mae_after = mean_absolute_error(actuals[drift_point:], predictions[drift_point:])
                performance_degradation = (mae_after - mae_before) / mae_before if mae_before > 0 else 0
            else:
                mae_before = mean_absolute_error(actuals, predictions)
                mae_after = mae_before
                performance_degradation = 0
            
            drift_info = {
                'variable': variable_name,
                'drift_type': DriftType.CONCEPT_DRIFT.value,
                'is_significant': is_drift,
                'drift_point': int(drift_point) if drift_point else None,
                'mae_before': float(mae_before),
                'mae_after': float(mae_after),
                'performance_degradation_pct': float(performance_degradation * 100),
                'severity': AlertSeverity.CRITICAL.value if is_drift else AlertSeverity.INFO.value
            }
            
            self._log_drift_detection(drift_info)
            
            if is_drift and performance_degradation > 0.2:  # Degradación > 20%
                self.logger.warning(f"⚠️ Concept drift detectado en {variable_name}: degradación {performance_degradation:.1%}")
                self._create_alert(
                    severity=AlertSeverity.CRITICAL,
                    category="drift",
                    title=f"Concept Drift Detectado: {variable_name}",
                    message=f"Performance degradada {performance_degradation:.1%} después del punto {drift_point}",
                    affected_variables=[variable_name],
                    metrics=drift_info,
                    recommended_action="Re-entrenar modelo URGENTE con datos recientes"
                )
            
            return drift_info
            
        except Exception as e:
            self.logger.error(f"Error detectando concept drift: {e}")
            return {'error': str(e)}
    
    def monitor_model_performance(self, 
                                 variable_name: str, 
                                 model_type: str,
                                 current_metrics: Dict[str, float],
                                 baseline_metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        Monitorea degradación de performance comparando con baseline.
        """
        performance_analysis = {
            'variable': variable_name,
            'model_type': model_type,
            'timestamp': datetime.now().isoformat(),
            'metrics_comparison': {}
        }
        
        alerts_triggered = []
        
        for metric_name in ['rmse', 'mae', 'r2']:
            if metric_name in current_metrics and metric_name in baseline_metrics:
                current_val = current_metrics[metric_name]
                baseline_val = baseline_metrics[metric_name]
                
                # Calcular degradación (menor es mejor para RMSE/MAE, mayor es mejor para R²)
                if metric_name in ['rmse', 'mae']:
                    degradation = (current_val - baseline_val) / baseline_val if baseline_val > 0 else 0
                else:  # R²
                    degradation = (baseline_val - current_val) / baseline_val if baseline_val > 0 else 0
                
                performance_analysis['metrics_comparison'][metric_name] = {
                    'current': float(current_val),
                    'baseline': float(baseline_val),
                    'degradation_pct': float(degradation * 100)
                }
                
                # Persistir métrica
                self._log_performance_metric(variable_name, model_type, metric_name, current_val, baseline_val)
                
                # Alerta si degradación > 15%
                if degradation > 0.15:
                    alerts_triggered.append({
                        'metric': metric_name,
                        'degradation': degradation
                    })
        
        if alerts_triggered:
            self._create_alert(
                severity=AlertSeverity.WARNING,
                category="performance",
                title=f"Degradación de Performance: {variable_name}",
                message=f"Modelo {model_type} muestra degradación en {len(alerts_triggered)} métricas",
                affected_variables=[variable_name],
                metrics=performance_analysis,
                recommended_action="Evaluar re-entrenamiento o ajuste de hiperparámetros"
            )
        
        return performance_analysis
    
    def _create_alert(self, 
                     severity: AlertSeverity, 
                     category: str,
                     title: str,
                     message: str,
                     affected_variables: List[str],
                     metrics: Dict[str, Any],
                     recommended_action: str):
        """Crea y registra una alerta"""
        alert = Alert(
            alert_id=f"alert_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            timestamp=datetime.now().isoformat(),
            severity=severity.value,
            category=category,
            title=title,
            message=message,
            affected_variables=affected_variables,
            metrics=metrics,
            recommended_action=recommended_action
        )
        
        self.alert_buffer.append(alert)
        self._persist_alert(alert)
        
        return alert
    
    def _persist_alert(self, alert: Alert):
        """Persiste alerta en DB"""
        with self.db.engine.connect() as conn:
            with conn.begin():
                conn.execute(text("""
                    INSERT INTO monitoring_events (
                        event_type, severity, affected_variables, metrics, details
                    ) VALUES ('alert', :severity, :vars, :metrics, :details)
                """), {
                    'severity': alert.severity,
                    'vars': str(alert.affected_variables),
                    'metrics': str(alert.metrics),
                    'details': f"{alert.title} | {alert.message} | Acción: {alert.recommended_action}"
                })
    
    def _log_drift_detection(self, drift_info: Dict[str, Any]):
        """Registra detección de drift en DB"""
        with self.db.engine.connect() as conn:
            with conn.begin():
                conn.execute(text("""
                    INSERT INTO monitoring_drift_detection (
                        variable_name, drift_type, drift_score, statistical_test, p_value, is_significant
                    ) VALUES (:var, :type, :score, :test, :pval, :sig)
                """), {
                    'var': drift_info.get('variable'),
                    'type': drift_info.get('drift_type'),
                    'score': drift_info.get('ks_statistic', 0.0),
                    'test': 'kolmogorov_smirnov',
                    'pval': drift_info.get('ks_pvalue', 1.0),
                    'sig': drift_info.get('is_significant', False)
                })
    
    def _log_performance_metric(self, 
                               variable_name: str, 
                               model_type: str,
                               metric_name: str,
                               metric_value: float,
                               baseline_value: float):
        """Registra métrica de performance en DB"""
        with self.db.engine.connect() as conn:
            with conn.begin():
                conn.execute(text("""
                    INSERT INTO monitoring_performance_metrics (
                        variable_name, model_type, metric_name, metric_value, baseline_value
                    ) VALUES (:var, :type, :name, :value, :baseline)
                """), {
                    'var': variable_name,
                    'type': model_type,
                    'name': metric_name,
                    'value': metric_value,
                    'baseline': baseline_value
                })
    
    def get_recent_alerts(self, hours: int = 24, severity: str = None) -> List[Alert]:
        """Obtiene alertas recientes"""
        return [a for a in self.alert_buffer if (not severity or a.severity == severity)]
    
    def generate_monitoring_report(self) -> str:
        """Genera reporte de estado del monitoreo"""
        report = []
        report.append("=" * 80)
        report.append("REPORTE DE MONITOREO CONTINUO - PROGNOSIS II")
        report.append(f"Generado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("=" * 80)
        report.append("")
        
        # Alertas activas
        critical_alerts = [a for a in self.alert_buffer if a.severity == AlertSeverity.CRITICAL.value]
        warning_alerts = [a for a in self.alert_buffer if a.severity == AlertSeverity.WARNING.value]
        
        report.append(f"1. ALERTAS ACTIVAS")
        report.append(f"   - Críticas: {len(critical_alerts)}")
        report.append(f"   - Advertencias: {len(warning_alerts)}")
        report.append("")
        
        if critical_alerts:
            report.append("2. ALERTAS CRÍTICAS")
            report.append("-" * 80)
            for alert in critical_alerts[:5]:  # Top 5
                report.append(f"   [{alert.timestamp}] {alert.title}")
                report.append(f"      → {alert.message}")
                report.append(f"      → Acción: {alert.recommended_action}")
                report.append("")
        
        report.append("=" * 80)
        return "\n".join(report)
