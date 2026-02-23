"""
✅ FASE 3 P2: Model Registry Completo
Sistema avanzado de versionado y gestión de modelos con trazabilidad completa.
"""

import logging
import json
import hashlib
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum
from sqlalchemy import text


class ModelStatus(Enum):
    """Estados del modelo en el registry"""
    TRAINING = "training"
    VALIDATION = "validation"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"
    FAILED = "failed"


@dataclass
class ModelVersion:
    """Versión de modelo con metadatos completos"""
    model_id: str
    version: str  # Semantic versioning: MAJOR.MINOR.PATCH
    variable_name: str
    model_type: str  # "sarimax", "prophet", "isolation_forest"
    status: str
    parameters: Dict[str, Any]
    performance_metrics: Dict[str, Any]
    training_data_hash: str
    created_at: str
    updated_at: str
    created_by: str = "prognosis_system"
    tags: List[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.metadata is None:
            self.metadata = {}


class ModelRegistry:
    """
    Registry completo de modelos con versionado semántico.
    Funcionalidades:
    - Versionado semántico automático (MAJOR.MINOR.PATCH)
    - Tracking de métricas y parámetros
    - Gestión de estados del modelo (staging → production)
    - Comparación de versiones
    - Rollback a versiones anteriores
    - Auditoría completa
    """
    
    def __init__(self, db_manager):
        self.db = db_manager
        self.logger = logging.getLogger(__name__)
        self._ensure_registry_schema()
        
    def _ensure_registry_schema(self):
        """Crea esquema de MLOps Registry si no existe"""
        with self.db.engine.connect() as conn:
            with conn.begin():
                # Tabla principal de versiones de modelos
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS mlops_model_registry (
                        model_id VARCHAR PRIMARY KEY,
                        version VARCHAR NOT NULL,
                        variable_name VARCHAR NOT NULL,
                        model_type VARCHAR NOT NULL,
                        status VARCHAR DEFAULT 'training',
                        parameters JSONB,
                        performance_metrics JSONB,
                        training_data_hash VARCHAR,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        created_by VARCHAR DEFAULT 'prognosis_system',
                        tags JSONB DEFAULT '[]',
                        metadata JSONB DEFAULT '{}'
                    );
                """))
                
                # Tabla de transiciones de estado
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS mlops_model_transitions (
                        transition_id SERIAL PRIMARY KEY,
                        model_id VARCHAR REFERENCES mlops_model_registry(model_id),
                        from_status VARCHAR,
                        to_status VARCHAR,
                        transitioned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        transitioned_by VARCHAR,
                        reason TEXT
                    );
                """))
                
                # Tabla de comparaciones de modelos
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS mlops_model_comparisons (
                        comparison_id SERIAL PRIMARY KEY,
                        model_id_a VARCHAR,
                        model_id_b VARCHAR,
                        comparison_metrics JSONB,
                        winner VARCHAR,
                        compared_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        compared_by VARCHAR
                    );
                """))
                
        self.logger.info("✅ MLOps Model Registry schema inicializado")
    
    def register_model(self, 
                      variable_name: str, 
                      model_type: str, 
                      parameters: Dict[str, Any],
                      performance_metrics: Dict[str, Any],
                      training_data: Any,
                      tags: List[str] = None,
                      metadata: Dict[str, Any] = None) -> ModelVersion:
        """
        Registra una nueva versión de modelo.
        El versionado es automático basado en cambios detectados.
        """
        # Calcular hash de datos de entrenamiento
        data_hash = self._calculate_data_hash(training_data)
        
        # Obtener última versión del modelo para esta variable
        last_version = self._get_latest_version(variable_name, model_type)
        
        # Determinar nueva versión semántica
        new_version = self._increment_version(
            last_version, 
            parameters, 
            performance_metrics, 
            data_hash
        )
        
        # Crear model_id único
        model_id = f"{variable_name}_{model_type}_{new_version}_{data_hash[:8]}"
        
        # Crear objeto ModelVersion
        model_version = ModelVersion(
            model_id=model_id,
            version=new_version,
            variable_name=variable_name,
            model_type=model_type,
            status=ModelStatus.TRAINING.value,
            parameters=parameters,
            performance_metrics=performance_metrics,
            training_data_hash=data_hash,
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
            tags=tags or [],
            metadata=metadata or {}
        )
        
        # Persistir en DB
        self._save_to_db(model_version)
        
        self.logger.info(f"✅ Modelo registrado: {model_id} (v{new_version})")
        return model_version
    
    def _calculate_data_hash(self, training_data: Any) -> str:
        """Calcula hash de datos de entrenamiento para detectar concept drift"""
        try:
            if hasattr(training_data, 'values'):
                data_bytes = training_data.values.tobytes()
            else:
                data_bytes = str(training_data).encode()
            
            return hashlib.sha256(data_bytes).hexdigest()
        except Exception:
            return hashlib.sha256(str(datetime.now()).encode()).hexdigest()
    
    def _get_latest_version(self, variable_name: str, model_type: str) -> Optional[Dict[str, Any]]:
        """Obtiene la última versión del modelo para una variable"""
        with self.db.engine.connect() as conn:
            result = conn.execute(text("""
                SELECT * FROM mlops_model_registry
                WHERE variable_name = :var AND model_type = :type
                ORDER BY created_at DESC
                LIMIT 1
            """), {'var': variable_name, 'type': model_type})
            
            row = result.fetchone()
            return dict(row._mapping) if row else None
    
    def _increment_version(self, 
                          last_version_data: Optional[Dict[str, Any]], 
                          new_params: Dict[str, Any],
                          new_metrics: Dict[str, Any],
                          new_data_hash: str) -> str:
        """
        Determina el incremento de versión semántica.
        Reglas:
        - MAJOR: Cambio en arquitectura del modelo o data hash (concept drift)
        - MINOR: Cambio en hiperparámetros
        - PATCH: Re-entrenamiento con mismos parámetros
        """
        if not last_version_data:
            return "1.0.0"
        
        last_version = last_version_data['version']
        major, minor, patch = map(int, last_version.split('.'))
        
        # MAJOR: Concept drift detectado
        if new_data_hash != last_version_data['training_data_hash']:
            return f"{major + 1}.0.0"
        
        # MINOR: Cambio en parámetros
        if new_params != last_version_data.get('parameters', {}):
            return f"{major}.{minor + 1}.0"
        
        # PATCH: Re-entrenamiento
        return f"{major}.{minor}.{patch + 1}"
    
    def _save_to_db(self, model_version: ModelVersion):
        """Persiste versión de modelo en DB"""
        with self.db.engine.connect() as conn:
            with conn.begin():
                conn.execute(text("""
                    INSERT INTO mlops_model_registry (
                        model_id, version, variable_name, model_type, status,
                        parameters, performance_metrics, training_data_hash,
                        created_at, updated_at, created_by, tags, metadata
                    ) VALUES (
                        :id, :version, :var, :type, :status,
                        :params, :metrics, :hash,
                        :created, :updated, :by, :tags, :meta
                    )
                    ON CONFLICT (model_id) DO UPDATE SET
                        updated_at = EXCLUDED.updated_at,
                        status = EXCLUDED.status
                """), {
                    'id': model_version.model_id,
                    'version': model_version.version,
                    'var': model_version.variable_name,
                    'type': model_version.model_type,
                    'status': model_version.status,
                    'params': json.dumps(model_version.parameters),
                    'metrics': json.dumps(model_version.performance_metrics),
                    'hash': model_version.training_data_hash,
                    'created': model_version.created_at,
                    'updated': model_version.updated_at,
                    'by': model_version.created_by,
                    'tags': json.dumps(model_version.tags),
                    'meta': json.dumps(model_version.metadata)
                })
    
    def transition_model(self, 
                        model_id: str, 
                        to_status: str, 
                        reason: str = "",
                        transitioned_by: str = "system") -> bool:
        """
        Cambia el estado de un modelo (e.g., staging → production).
        Registra la transición para auditoría.
        """
        with self.db.engine.connect() as conn:
            with conn.begin():
                # Obtener estado actual
                result = conn.execute(text("""
                    SELECT status FROM mlops_model_registry WHERE model_id = :id
                """), {'id': model_id})
                
                row = result.fetchone()
                if not row:
                    self.logger.error(f"Modelo {model_id} no encontrado")
                    return False
                
                from_status = row[0]
                
                # Actualizar estado
                conn.execute(text("""
                    UPDATE mlops_model_registry
                    SET status = :status, updated_at = CURRENT_TIMESTAMP
                    WHERE model_id = :id
                """), {'status': to_status, 'id': model_id})
                
                # Registrar transición
                conn.execute(text("""
                    INSERT INTO mlops_model_transitions (
                        model_id, from_status, to_status, transitioned_by, reason
                    ) VALUES (:id, :from, :to, :by, :reason)
                """), {
                    'id': model_id,
                    'from': from_status,
                    'to': to_status,
                    'by': transitioned_by,
                    'reason': reason
                })
        
        self.logger.info(f"✅ Modelo {model_id}: {from_status} → {to_status}")
        return True
    
    def get_production_model(self, variable_name: str, model_type: str) -> Optional[Dict[str, Any]]:
        """Obtiene el modelo actualmente en producción para una variable"""
        with self.db.engine.connect() as conn:
            result = conn.execute(text("""
                SELECT * FROM mlops_model_registry
                WHERE variable_name = :var 
                  AND model_type = :type 
                  AND status = 'production'
                ORDER BY updated_at DESC
                LIMIT 1
            """), {'var': variable_name, 'type': model_type})
            
            row = result.fetchone()
            return dict(row._mapping) if row else None
    
    def compare_models(self, 
                      model_id_a: str, 
                      model_id_b: str,
                      comparison_metric: str = 'rmse') -> Dict[str, Any]:
        """
        Compara dos versiones de modelos y determina el ganador.
        Registra la comparación para auditoría.
        """
        with self.db.engine.connect() as conn:
            # Obtener modelos
            result_a = conn.execute(text("""
                SELECT * FROM mlops_model_registry WHERE model_id = :id
            """), {'id': model_id_a})
            
            result_b = conn.execute(text("""
                SELECT * FROM mlops_model_registry WHERE model_id = :id
            """), {'id': model_id_b})
            
            model_a = dict(result_a.fetchone()._mapping) if result_a else None
            model_b = dict(result_b.fetchone()._mapping) if result_b else None
            
            if not model_a or not model_b:
                return {'error': 'Uno o ambos modelos no encontrados'}
            
            # Comparar métricas
            metrics_a = model_a.get('performance_metrics', {})
            metrics_b = model_b.get('performance_metrics', {})
            
            metric_a = metrics_a.get(comparison_metric, float('inf'))
            metric_b = metrics_b.get(comparison_metric, float('inf'))
            
            # Determinar ganador (menor es mejor para RMSE/MAE)
            winner = model_id_a if metric_a < metric_b else model_id_b
            
            comparison_result = {
                'model_a': model_id_a,
                'model_b': model_id_b,
                'metric': comparison_metric,
                'value_a': metric_a,
                'value_b': metric_b,
                'winner': winner,
                'improvement_pct': abs((metric_a - metric_b) / metric_b) * 100 if metric_b != 0 else 0
            }
            
            # Registrar comparación
            with conn.begin():
                conn.execute(text("""
                    INSERT INTO mlops_model_comparisons (
                        model_id_a, model_id_b, comparison_metrics, winner, compared_by
                    ) VALUES (:a, :b, :metrics, :winner, :by)
                """), {
                    'a': model_id_a,
                    'b': model_id_b,
                    'metrics': json.dumps(comparison_result),
                    'winner': winner,
                    'by': 'system'
                })
            
            return comparison_result
