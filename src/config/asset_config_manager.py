"""
✅ FASE 3 P2: Asset Configuration Manager
Sistema metadata-driven para agnosticismo completo.
Permite configurar el sistema para cualquier tipo de activo industrial.
"""

import json
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum


class AssetType(Enum):
    """Tipos de activos soportados (extensible)"""
    SUBSTATION = "substation"
    PUMP = "pump"
    TURBINE = "turbine"
    COMPRESSOR = "compressor"
    MOTOR = "motor"
    HVAC = "hvac"
    GENERIC = "generic"


class TemporalGranularity(Enum):
    """Granularidad temporal de los datos"""
    SECONDS = "seconds"
    MINUTES = "minutes"
    HOURS = "hours"
    DAYS = "days"


@dataclass
class VariableMetadata:
    """Metadatos de una variable del activo"""
    name: str
    unit: str
    expected_range: tuple  # (min, max)
    critical_thresholds: Dict[str, float]  # {'warning': X, 'critical': Y}
    sampling_frequency: str  # e.g., "1H", "15min"
    data_type: str  # "numeric", "categorical", "binary"
    is_mandatory: bool = True
    description: str = ""


@dataclass
class AssetConfiguration:
    """Configuración completa de un tipo de activo"""
    asset_type: str
    asset_name: str
    temporal_granularity: str
    expected_seasonality_periods: List[int]  # e.g., [24, 168] para diario y semanal
    min_data_points_for_training: int
    variables: List[VariableMetadata]
    failure_modes: List[Dict[str, Any]]  # Modos de falla conocidos del activo
    business_rules: Dict[str, Any]  # Reglas específicas del negocio
    alert_configuration: Dict[str, Any]
    metadata: Dict[str, Any]  # Información adicional


class AssetConfigManager:
    """
    Gestor de configuraciones de activos.
    Permite cargar, guardar y validar configuraciones metadata-driven.
    """
    
    def __init__(self, config_dir: str = "configs/assets/"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        self.active_config: Optional[AssetConfiguration] = None
        
    def load_configuration(self, asset_type: str, asset_name: str = None) -> AssetConfiguration:
        """
        Carga configuración de un tipo de activo desde archivo JSON.
        Args:
            asset_type: Tipo de activo (e.g., "substation", "pump")
            asset_name: Nombre específico del activo (opcional)
        """
        # Intentar cargar configuración específica del activo
        if asset_name:
            config_file = self.config_dir / f"{asset_type}_{asset_name}.json"
        else:
            config_file = self.config_dir / f"{asset_type}_template.json"
        
        if not config_file.exists():
            self.logger.warning(f"Configuración {config_file} no encontrada. Usando configuración genérica.")
            return self._create_generic_config(asset_type, asset_name)
        
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
            
            # Reconstruir objetos VariableMetadata
            variables = [
                VariableMetadata(**var_dict) for var_dict in config_dict.get('variables', [])
            ]
            
            config = AssetConfiguration(
                asset_type=config_dict['asset_type'],
                asset_name=config_dict['asset_name'],
                temporal_granularity=config_dict['temporal_granularity'],
                expected_seasonality_periods=config_dict.get('expected_seasonality_periods', [24]),
                min_data_points_for_training=config_dict.get('min_data_points_for_training', 100),
                variables=variables,
                failure_modes=config_dict.get('failure_modes', []),
                business_rules=config_dict.get('business_rules', {}),
                alert_configuration=config_dict.get('alert_configuration', {}),
                metadata=config_dict.get('metadata', {})
            )
            
            self.active_config = config
            self.logger.info(f"✅ Configuración cargada: {asset_type}/{asset_name}")
            return config
            
        except Exception as e:
            self.logger.error(f"Error cargando configuración: {e}")
            return self._create_generic_config(asset_type, asset_name)
    
    def save_configuration(self, config: AssetConfiguration):
        """Guarda configuración de activo en archivo JSON"""
        config_file = self.config_dir / f"{config.asset_type}_{config.asset_name}.json"
        
        try:
            # Convertir a dict serializable
            config_dict = {
                'asset_type': config.asset_type,
                'asset_name': config.asset_name,
                'temporal_granularity': config.temporal_granularity,
                'expected_seasonality_periods': config.expected_seasonality_periods,
                'min_data_points_for_training': config.min_data_points_for_training,
                'variables': [asdict(var) for var in config.variables],
                'failure_modes': config.failure_modes,
                'business_rules': config.business_rules,
                'alert_configuration': config.alert_configuration,
                'metadata': config.metadata
            }
            
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"✅ Configuración guardada: {config_file}")
            
        except Exception as e:
            self.logger.error(f"Error guardando configuración: {e}")
    
    def _create_generic_config(self, asset_type: str, asset_name: str = None) -> AssetConfiguration:
        """Crea configuración genérica por defecto"""
        return AssetConfiguration(
            asset_type=asset_type,
            asset_name=asset_name or "generic_asset",
            temporal_granularity="hours",
            expected_seasonality_periods=[24, 168],  # Diario y semanal por defecto
            min_data_points_for_training=100,
            variables=[],  # Se detectarán automáticamente
            failure_modes=[],
            business_rules={},
            alert_configuration={
                'warning_threshold': 0.6,
                'critical_threshold': 0.8
            },
            metadata={}
        )
    
    def create_substation_template(self, substation_name: str) -> AssetConfiguration:
        """Crea template específico para subestaciones eléctricas"""
        variables = [
            VariableMetadata(
                name="voltage_l1",
                unit="V",
                expected_range=(0, 500),
                critical_thresholds={'warning': 460, 'critical': 480},
                sampling_frequency="1H",
                data_type="numeric",
                description="Voltaje línea 1"
            ),
            VariableMetadata(
                name="current_l1",
                unit="A",
                expected_range=(0, 1000),
                critical_thresholds={'warning': 900, 'critical': 950},
                sampling_frequency="1H",
                data_type="numeric",
                description="Corriente línea 1"
            ),
            VariableMetadata(
                name="temperature_transformer",
                unit="°C",
                expected_range=(-10, 100),
                critical_thresholds={'warning': 80, 'critical': 90},
                sampling_frequency="1H",
                data_type="numeric",
                description="Temperatura del transformador"
            ),
        ]
        
        failure_modes = [
            {
                'name': 'overheating',
                'indicators': ['temperature_transformer'],
                'threshold': 90,
                'criticality': 'high'
            },
            {
                'name': 'overload',
                'indicators': ['current_l1', 'current_l2', 'current_l3'],
                'threshold': 950,
                'criticality': 'critical'
            },
            {
                'name': 'voltage_instability',
                'indicators': ['voltage_l1', 'voltage_l2', 'voltage_l3'],
                'threshold_std': 50,
                'criticality': 'medium'
            }
        ]
        
        config = AssetConfiguration(
            asset_type=AssetType.SUBSTATION.value,
            asset_name=substation_name,
            temporal_granularity=TemporalGranularity.HOURS.value,
            expected_seasonality_periods=[24, 168],  # Diario y semanal
            min_data_points_for_training=240,  # 10 días mínimo
            variables=variables,
            failure_modes=failure_modes,
            business_rules={
                'maintenance_window_hours': 24,
                'emergency_response_time_hours': 2
            },
            alert_configuration={
                'warning_threshold': 0.6,
                'critical_threshold': 0.8,
                'notification_channels': ['email', 'sms', 'dashboard']
            },
            metadata={
                'location': 'TBD',
                'commissioning_date': 'TBD',
                'manufacturer': 'TBD'
            }
        )
        
        return config
    
    def get_variable_config(self, variable_name: str) -> Optional[VariableMetadata]:
        """Obtiene configuración de una variable específica"""
        if not self.active_config:
            return None
        
        for var in self.active_config.variables:
            if var.name == variable_name:
                return var
        return None
    
    def validate_data_against_config(self, data: Dict[str, Any]) -> Dict[str, List[str]]:
        """
        Valida datos de entrada contra la configuración activa.
        Returns:
            Dict con 'errors' y 'warnings'
        """
        if not self.active_config:
            return {'errors': ['No hay configuración activa'], 'warnings': []}
        
        errors = []
        warnings = []
        
        # Verificar variables obligatorias
        for var in self.active_config.variables:
            if var.is_mandatory and var.name not in data:
                errors.append(f"Variable obligatoria faltante: {var.name}")
        
        # Verificar rangos esperados
        for var in self.active_config.variables:
            if var.name in data:
                value = data[var.name]
                min_val, max_val = var.expected_range
                
                if not (min_val <= value <= max_val):
                    warnings.append(
                        f"{var.name}={value} fuera de rango esperado [{min_val}, {max_val}]"
                    )
        
        return {'errors': errors, 'warnings': warnings}
