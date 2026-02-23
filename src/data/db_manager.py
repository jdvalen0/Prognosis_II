import logging
import pandas as pd
import numpy as np
import json
from sqlalchemy import create_engine, text
from typing import Dict, Any, Tuple
import unicodedata
from psycopg2 import connect
from datetime import datetime
from src.config import SystemConfig

class DatabaseManager:
    """Gestiona la conexión y operaciones básicas de la base de datos."""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.engine = self._create_engine()
        self._setup_logging()

    def _setup_logging(self):
        self.logger = logging.getLogger(__name__)

    def ensure_database_exists(self):
        """Asegura que la base de datos objetivo exista."""
        if getattr(self.config, "USE_SQLITE", False) or "sqlite" in (self.config.DATABASE_URL or ""):
            return
        try:
            conn = connect(
                user=self.config.DB_USER,
                password=self.config.DB_PASSWORD,
                host=self.config.DB_HOST,
                port=self.config.DB_PORT,
                database="postgres"
            )
            conn.set_isolation_level(0)
            cursor = conn.cursor()
            cursor.execute(f"SELECT 1 FROM pg_database WHERE datname = '{self.config.DB_NAME}'")
            exists = cursor.fetchone()
            if not exists:
                self.logger.info(f"Creando base de datos {self.config.DB_NAME}...")
                cursor.execute(f'CREATE DATABASE "{self.config.DB_NAME}"')
            cursor.close()
            conn.close()
        except Exception as e:
            self.logger.warning(f"No se pudo asegurar la existencia de la DB (puede que ya exista): {e}")

    def _create_engine(self):
        try:
            engine = create_engine(self.config.DATABASE_URL)
            return engine
        except Exception as e:
            logging.error(f"Error al crear motor de base de datos: {e}")
            raise

    def _is_sqlite(self) -> bool:
        return self.engine.dialect.name == "sqlite"

    def setup_mlops_schema(self):
        """Crea el esquema completo de MLOps alineado con el estándar de Diamante."""
        try:
            with self.engine.connect() as conn:
                with conn.begin():
                    if self._is_sqlite():
                        self._setup_mlops_schema_sqlite(conn)
                    else:
                        self._setup_mlops_schema_pg(conn)
            self.logger.info("Esquema MLOps (Diamante) configurado exitosamente.")
        except Exception as e:
            self.logger.error(f"Error configurando esquema MLOps: {e}")

    def _setup_mlops_schema_sqlite(self, conn):
        """Esquema MLOps compatible con SQLite (TEXT/INTEGER)."""
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS baseline_models (
                variable_name TEXT PRIMARY KEY,
                model_type TEXT DEFAULT 'ensemble',
                baseline_stats TEXT,
                model_parameters TEXT,
                confidence_intervals TEXT,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS baseline_model_versions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                variable_name TEXT NOT NULL,
                version INTEGER NOT NULL,
                model_parameters TEXT,
                performance_metrics TEXT,
                is_active INTEGER DEFAULT 1,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(variable_name, version)
            );
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS baseline_incremental_control (
                variable_name TEXT PRIMARY KEY,
                last_processed_timestamp TEXT,
                batch_count INTEGER DEFAULT 0,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS pattern_registry (
                variable_name TEXT PRIMARY KEY,
                has_seasonality INTEGER DEFAULT 0,
                seasonal_period INTEGER DEFAULT 0,
                stationarity_status TEXT,
                detected_patterns TEXT,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS variable_analysis_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                variable_name TEXT NOT NULL,
                variance_score REAL,
                stability_score REAL,
                trend_score REAL,
                correlation_score REAL,
                final_score REAL,
                category TEXT,
                analyzed_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS scaler_parameters (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                scaler_type TEXT NOT NULL DEFAULT 'StandardScaler',
                mean_params TEXT NOT NULL,
                scale_params TEXT NOT NULL,
                columns_list TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                is_active INTEGER DEFAULT 1
            );
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS variable_mapping (
                normalized_name TEXT PRIMARY KEY,
                original_name TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
        """))

    def _setup_mlops_schema_pg(self, conn):
        """Esquema MLOps para PostgreSQL (JSONB/SERIAL)."""
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS baseline_models (
                variable_name VARCHAR PRIMARY KEY,
                model_type VARCHAR DEFAULT 'ensemble',
                baseline_stats JSONB,
                model_parameters JSONB,
                confidence_intervals JSONB,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS baseline_model_versions (
                id SERIAL PRIMARY KEY,
                variable_name VARCHAR NOT NULL,
                version INTEGER NOT NULL,
                model_parameters JSONB,
                performance_metrics JSONB,
                is_active BOOLEAN DEFAULT true,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(variable_name, version)
            );
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS baseline_incremental_control (
                variable_name VARCHAR PRIMARY KEY,
                last_processed_timestamp TIMESTAMP,
                batch_count INTEGER DEFAULT 0,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS pattern_registry (
                variable_name VARCHAR PRIMARY KEY,
                has_seasonality BOOLEAN,
                seasonal_period INTEGER DEFAULT 0,
                stationarity_status VARCHAR,
                detected_patterns JSONB,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """))
        try:
            conn.execute(text("""
                DO $$
                BEGIN
                    IF NOT EXISTS (
                        SELECT 1 FROM information_schema.columns
                        WHERE table_name = 'pattern_registry' AND column_name = 'seasonal_period'
                    ) THEN
                        ALTER TABLE pattern_registry ADD COLUMN seasonal_period INTEGER DEFAULT 0;
                    END IF;
                END $$;
            """))
        except Exception as mig_error:
            self.logger.warning(f"Migración de seasonal_period: {mig_error}")

    def check_connection(self) -> bool:
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except Exception as e:
            self.logger.error(f"Error de conexión: {e}")
            return False

    def get_baseline_stats(self, variable_name: str) -> Dict[str, float]:
        """Obtiene las estadísticas del modelo base actual para aprendizaje incremental."""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(
                    text("SELECT baseline_stats FROM baseline_models WHERE variable_name = :var"),
                    {'var': variable_name}
                ).fetchone()
                
                if result and result[0]:
                    import json
                    return json.loads(result[0]) if isinstance(result[0], str) else result[0]
            return {}
        except Exception as e:
            self.logger.error(f"Error recuperando estadísticas base: {e}")
            return {}
    def save_selection_metrics(self, metrics: Dict[str, Dict[str, Any]]):
        """Persiste las métricas de selección de variables para auditoría."""
        try:
            with self.engine.connect() as conn:
                with conn.begin():
                    if not self._is_sqlite():
                        conn.execute(text("""
                            CREATE TABLE IF NOT EXISTS variable_analysis_history (
                                id SERIAL PRIMARY KEY,
                                variable_name VARCHAR NOT NULL,
                                variance_score FLOAT,
                                stability_score FLOAT,
                                trend_score FLOAT,
                                correlation_score FLOAT,
                                final_score FLOAT,
                                category VARCHAR,
                                analyzed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                            );
                        """))
                    for var, m in metrics.items():
                        conn.execute(text("""
                            INSERT INTO variable_analysis_history 
                            (variable_name, variance_score, stability_score, trend_score, correlation_score, final_score, category)
                            VALUES (:var, :var_s, :st_s, :tr_s, :co_s, :fi_s, :cat)
                        """), {
                            'var': var,
                            'var_s': m['variance'],
                            'st_s': m['stability'],
                            'tr_s': m['trend'],
                            'co_s': m['correlation'],
                            'fi_s': m['final_score'],
                            'cat': m['category']
                        })
            self.logger.info("Métricas de selección persistidas en el historial.")
        except Exception as e:
            self.logger.error(f"Error persistiendo métricas de selección: {e}")

    def save_model_version(self, variable: str, params: Dict[str, Any], metrics: Dict[str, Any]):
        """Registra una nueva versión del modelo para trazabilidad MLOps."""
        try:
            with self.engine.connect() as conn:
                with conn.begin():
                    # Obtener última versión
                    res = conn.execute(text("""
                        SELECT COALESCE(MAX(version), 0) FROM baseline_model_versions 
                        WHERE variable_name = :var
                    """), {'var': variable}).fetchone()
                    new_version = res[0] + 1
                    
                    # Desactivar versiones anteriores
                    conn.execute(text("""
                        UPDATE baseline_model_versions SET is_active = false 
                        WHERE variable_name = :var
                    """), {'var': variable})
                    
                    # Insertar nueva versión (is_active=1 compatible PostgreSQL/SQLite)
                    conn.execute(text("""
                        INSERT INTO baseline_model_versions 
                        (variable_name, version, model_parameters, performance_metrics, is_active)
                        VALUES (:var, :ver, :params, :met, 1)
                    """), {
                        'var': variable,
                        'ver': new_version,
                        'params': json.dumps(params),
                        'met': json.dumps(metrics)
                    })
            self.logger.info(f"Version {new_version} del modelo {variable} persistida.")
        except Exception as e:
            self.logger.error(f"Error salvando versión de modelo: {e}")

    def get_last_processed_timestamp(self, variable_name: str) -> datetime:
        """Obtiene el último timestamp procesado para un batch industrial."""
        try:
            with self.engine.connect() as conn:
                res = conn.execute(text("""
                    SELECT last_processed_timestamp FROM baseline_incremental_control 
                    WHERE variable_name = :var
                """), {'var': variable_name}).fetchone()
                return res[0] if res and res[0] else datetime(1970, 1, 1)
        except Exception as e:
            self.logger.error(f"Error obteniendo last_timestamp: {e}")
            return datetime(1970, 1, 1)

    def update_batch_control(self, variable_name: str, last_ts: datetime):
        """Actualiza el control de batches tras procesar nuevos datos."""
        try:
            with self.engine.connect() as conn:
                with conn.begin():
                    conn.execute(text("""
                        INSERT INTO baseline_incremental_control (variable_name, last_processed_timestamp, batch_count)
                        VALUES (:var, :ts, 1)
                        ON CONFLICT (variable_name) DO UPDATE SET 
                            last_processed_timestamp = EXCLUDED.last_processed_timestamp,
                            batch_count = baseline_incremental_control.batch_count + 1,
                            updated_at = CURRENT_TIMESTAMP
                    """), {'var': variable_name, 'ts': last_ts})
        except Exception as e:
            self.logger.error(f"Error actualizando control de batch: {e}")
