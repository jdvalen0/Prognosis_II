import logging
import pandas as pd
import numpy as np
import unicodedata
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict, Any
from sqlalchemy import text
from datetime import datetime
import json
from src.data.db_manager import DatabaseManager

class DataPreprocessor:
    """
    Preprocesador robusto para datos industriales agnósticos.
    
    ✅ CORRECCIÓN CRÍTICA: Usa StandardScaler (Z-score) como especifica la teoría y el notebook.
    Implementa fit/transform separado con persistencia de parámetros para evitar data leakage.
    """
    
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
        self.scaler = StandardScaler()  # ✅ Z-score: (x-μ)/σ según teoría
        self.scaler_fitted = False
        self.scaler_params = {}
        self.quality_report = {
            'missing_values': {},
            'irrelevant_columns': [],
            'data_types': {},
            'variable_mapping': {},
            'constant_variables': []
        }
        self._setup_logging()

    def _setup_logging(self):
        self.logger = logging.getLogger(__name__)

    def _normalize_column_name(self, column: str) -> str:
        """Normaliza el nombre de la columna para SQL y trazabilidad."""
        normalized = column.lower()
        normalized = unicodedata.normalize('NFKD', normalized)
        normalized = "".join([c for c in normalized if not unicodedata.combining(c)])
        normalized = ''.join(c if c.isalnum() else '_' for c in normalized)
        normalized = '_'.join(filter(None, normalized.split('_')))
        self.quality_report['variable_mapping'][normalized] = column
        return normalized

    def load_data(self, file_path: str, since_timestamp: datetime = None) -> pd.DataFrame:
        """Carga datos desde Excel o CSV, preparado para filtrado incremental (since_timestamp)."""
        try:
            self.logger.info(f"Cargando datos desde {file_path}...")
            if file_path.endswith('.xlsx'):
                data = pd.read_excel(file_path)
            else:
                data = pd.read_csv(file_path, low_memory=False)
            
            # Normalizar nombres inmediatamente para encontrar la fecha
            data.columns = [self._normalize_column_name(col) for col in data.columns]
            
            # Identificar columna temporal activa
            date_col = None
            for col in ['fecha', 'timestamp', 'date', 'datetime']:
                if col in data.columns:
                    date_col = col
                    break
            
            if date_col:
                data[date_col] = pd.to_datetime(data[date_col])
                if since_timestamp:
                    data = data[data[date_col] > since_timestamp]
                    self.logger.info(f"Filtrado incremental aplicado. Nuevos registros: {len(data)}")
            
            return data
        except Exception as e:
            self.logger.error(f"Error cargando datos: {e}")
            return pd.DataFrame()

    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Limpia y valida la estructura de los datos."""
        df = data.copy()
        df.columns = [self._normalize_column_name(col) for col in df.columns]
        
        # Identificar variables constantes
        for col in df.columns:
            unique_count = df[col].nunique()
            if unique_count <= 1:
                self.quality_report['constant_variables'].append({
                    'variable': col,
                    'value': df[col].iloc[0] if unique_count == 1 else None
                })
                self.quality_report['irrelevant_columns'].append(col)
        
        # Eliminar constantes
        df.drop(columns=self.quality_report['irrelevant_columns'], inplace=True)
        
        # Manejo de nulos (Mediana para robustez)
        for col in df.select_dtypes(include=[np.number]).columns:
            missing = df[col].isnull().sum()
            self.quality_report['missing_values'][col] = int(missing)
            if missing > 0:
                df[col].fillna(df[col].median(), inplace=True)
        
        return df

    def fit_scaler(self, training_data: pd.DataFrame):
        """
        ✅ CORRECCIÓN CRÍTICA: Fit scaler SOLO con datos de entrenamiento.
        Guarda parámetros (μ, σ) para reproducibilidad según teoría.
        """
        if training_data.empty:
            self.logger.warning("DataFrame vacío recibido para fit del scaler.")
            return
        
        numeric_cols = training_data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            self.logger.warning("No hay columnas numéricas para fit del scaler.")
            return
        
        # Excluir timestamp/fecha del escalado
        exclude_cols = ['timestamp', 'fecha', 'date', 'datetime']
        numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        if len(numeric_cols) == 0:
            self.logger.warning("No quedaron columnas numéricas después de excluir temporales.")
            return
        
        self.logger.info(f"Fitting StandardScaler con {len(numeric_cols)} columnas numéricas...")
        self.scaler.fit(training_data[numeric_cols])
        self.scaler_fitted = True
        
        # Guardar parámetros para reproducibilidad
        self.scaler_params = {
            'mean': self.scaler.mean_.tolist(),
            'scale': self.scaler.scale_.tolist(),
            'columns': numeric_cols
        }
        
        # Persistir en DB
        self._save_scaler_to_db()
        self.logger.info("✅ StandardScaler fitted y parámetros guardados para reproducibilidad.")
    
    def normalize_data(self, data: pd.DataFrame, fit_mode: bool = False) -> pd.DataFrame:
        """
        ✅ CORRECCIÓN CRÍTICA: Normaliza usando StandardScaler (Z-score).
        
        Args:
            data: DataFrame a normalizar
            fit_mode: Si True, hace fit del scaler (SOLO usar en entrenamiento inicial).
                     Si False, usa scaler pre-entrenado (producción/incremental).
        
        Returns:
            DataFrame normalizado con Z-score: (x-μ)/σ
        """
        if data.empty:
            self.logger.warning("DataFrame vacío recibido para normalización.")
            return data
        
        df = data.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        # Excluir timestamp/fecha del escalado
        exclude_cols = ['timestamp', 'fecha', 'date', 'datetime']
        numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        if len(numeric_cols) == 0:
            self.logger.warning("No hay columnas numéricas para normalizar.")
            return df
        
        # ✅ MODO ENTRENAMIENTO: fit_transform (solo primera vez)
        if fit_mode:
            self.logger.info("MODO ENTRENAMIENTO: Fitting StandardScaler con datos actuales")
            self.fit_scaler(df)
            df[numeric_cols] = self.scaler.transform(df[numeric_cols])
        
        # ✅ MODO PRODUCCIÓN: solo transform (usar parámetros guardados)
        else:
            # Intentar cargar scaler si no está fitted
            if not self.scaler_fitted:
                loaded = self._load_scaler_from_db()
                if not loaded:
                    raise RuntimeError(
                        "Scaler no está fitted y no se pudo cargar de DB. "
                        "Ejecute fit_scaler() primero o llame con fit_mode=True"
                    )
            
            self.logger.info("MODO PRODUCCIÓN: Usando StandardScaler pre-entrenado (Z-score)")
            df[numeric_cols] = self.scaler.transform(df[numeric_cols])
        
        return df
    
    def _save_scaler_to_db(self):
        """Guarda parámetros del scaler en DB para reproducibilidad."""
        try:
            with self.db.engine.connect() as conn:
                with conn.begin():
                    if self.db.engine.dialect.name != "sqlite":
                        conn.execute(text("""
                            CREATE TABLE IF NOT EXISTS scaler_parameters (
                                id SERIAL PRIMARY KEY,
                                scaler_type VARCHAR NOT NULL DEFAULT 'StandardScaler',
                                mean_params JSONB NOT NULL,
                                scale_params JSONB NOT NULL,
                                columns_list JSONB NOT NULL,
                                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                                is_active BOOLEAN DEFAULT true
                            );
                        """))
                    # Desactivar versiones anteriores
                    conn.execute(text("""
                        UPDATE scaler_parameters SET is_active = false
                        WHERE is_active = true;
                    """))
                    
                    # Insertar nueva versión
                    conn.execute(text("""
                        INSERT INTO scaler_parameters 
                        (scaler_type, mean_params, scale_params, columns_list)
                        VALUES (:type, :mean, :scale, :cols)
                    """), {
                        'type': 'StandardScaler',
                        'mean': json.dumps(self.scaler_params['mean']),
                        'scale': json.dumps(self.scaler_params['scale']),
                        'cols': json.dumps(self.scaler_params['columns'])
                    })
            
            self.logger.info("✅ Parámetros del scaler guardados en DB.")
        except Exception as e:
            self.logger.error(f"Error guardando parámetros del scaler: {e}")
    
    def _load_scaler_from_db(self) -> bool:
        """Carga parámetros del scaler desde DB."""
        try:
            with self.db.engine.connect() as conn:
                result = conn.execute(text("""
                    SELECT mean_params, scale_params, columns_list
                    FROM scaler_parameters
                    WHERE is_active = true
                    ORDER BY created_at DESC
                    LIMIT 1;
                """))
                
                row = result.fetchone()
                if row:
                    # Robust JSON handling: Check if already deserialized (list/dict) or string
                    mean_params = row[0] if not isinstance(row[0], str) else json.loads(row[0])
                    scale_params = row[1] if not isinstance(row[1], str) else json.loads(row[1])
                    columns_list = row[2] if not isinstance(row[2], str) else json.loads(row[2])
                    
                    # Reconstruir scaler
                    from sklearn.preprocessing import StandardScaler
                    self.scaler = StandardScaler()
                    self.scaler.mean_ = np.array(mean_params)
                    self.scaler.scale_ = np.array(scale_params)
                    self.scaler.n_features_in_ = len(columns_list)
                    self.scaler.feature_names_in_ = np.array(columns_list)
                    self.scaler_fitted = True
                    self.scaler_params = {
                        'mean': mean_params,
                        'scale': scale_params,
                        'columns': columns_list
                    }
                    
                    self.logger.info("✅ Parámetros del scaler cargados desde DB.")
                    return True
                else:
                    self.logger.warning("No se encontraron parámetros del scaler en DB.")
                    return False
        except Exception as e:
            self.logger.error(f"Error cargando parámetros del scaler: {e}")
            return False

    def save_to_db(self, df: pd.DataFrame, table_name: str):
        """Guarda datos y mapeo en DB (PostgreSQL o SQLite)."""
        with self.db.engine.connect() as conn:
            with conn.begin():
                if self.db.engine.dialect.name != "sqlite":
                    conn.execute(text("""
                        CREATE TABLE IF NOT EXISTS variable_mapping (
                            normalized_name VARCHAR PRIMARY KEY,
                            original_name VARCHAR NOT NULL,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        );
                    """))
                
                for norm, orig in self.quality_report['variable_mapping'].items():
                    conn.execute(text("""
                        INSERT INTO variable_mapping (normalized_name, original_name)
                        VALUES (:norm, :orig)
                        ON CONFLICT (normalized_name) DO UPDATE SET original_name = EXCLUDED.original_name
                    """), {'norm': norm, 'orig': orig})
                
                df.to_sql(table_name, conn, if_exists='replace', index=False)
        self.logger.info(f"Datos guardados en tabla {table_name}")
