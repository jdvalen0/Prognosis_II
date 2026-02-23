import logging
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from sqlalchemy import text
from src.config import SystemConfig
from src.data.db_manager import DatabaseManager
from src.data.preprocessor import DataPreprocessor
from src.features.selector import KeyVariableSelector
from src.models.baseline_modeler import BaselineModeler
from src.models.predictor import IndustrialFailurePredictor

# Configuración de logging global
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("PrognosisEngine")

class PrognosisEngine:
    """Motor principal de Prognosis II, capaz de operar en modo Batch Industrial."""
    
    def __init__(self):
        self.config = SystemConfig()
        self.db = DatabaseManager(self.config)
        
        # Inicializar infraestructura
        self.db.ensure_database_exists()
        self.db.setup_mlops_schema()
        
        if not self.db.check_connection():
            logger.error("No se pudo conectar a la base de datos industrial.")
            raise ConnectionError("Database connection failed.")

        self.preprocessor = DataPreprocessor(self.db)
        self.selector = KeyVariableSelector(self.config, self.db)
        self.modeler = BaselineModeler(self.config, self.db)
        self.predictor = IndustrialFailurePredictor(self.config, self.modeler)

    def run_pipeline(self, file_path: str):
        """
        Ejecución completa (Full Reset) - Útil para calibración inicial.
        
        ✅ CORRECCIÓN CRÍTICA: Flujo corregido para alinearse con notebook:
        1. Load → Clean → Normalize (fit) → Save → Select (sobre datos normalizados) → Model → Predict
        """
        try:
            logger.info(f"=== Iniciando Pipeline Full Run: {file_path} ===")
            
            # 1. Carga y Limpieza
            raw_data = self.preprocessor.load_data(file_path)
            clean_data = self.preprocessor.clean_data(raw_data)
            
            # 2. ✅ Normalización PRIMERO (fit_mode=True para entrenamiento inicial)
            normalized_data = self.preprocessor.normalize_data(clean_data, fit_mode=True)
            self.preprocessor.save_to_db(normalized_data, "normalized_data_table")
            
            # 3. ✅ Selección de Variables SOBRE DATOS NORMALIZADOS (como notebook)
            critical_vars = self.selector.select_critical_variables(normalized_data)

            # 4. Entrenamiento de Línea Base
            for var in critical_vars:
                # ✅ Agnóstico temporal: usar timestamp real como índice si existe
                if 'timestamp' in normalized_data.columns:
                    series_for_model = normalized_data.set_index(pd.to_datetime(normalized_data['timestamp']))[var]
                elif 'fecha' in normalized_data.columns:
                    series_for_model = normalized_data.set_index(pd.to_datetime(normalized_data['fecha']))[var]
                else:
                    series_for_model = normalized_data[var]

                results = self.modeler.fit_ensemble(var, series_for_model)
                self.modeler.save_baseline(results)

            # 5. Inferencia
            prediction = self.predictor.predict(normalized_data)
            self._report_results(prediction)
            
            return prediction

        except Exception as e:
            logger.error(f"Error en el pipeline full run: {e}", exc_info=True)

    def _load_full_historical_data(self) -> pd.DataFrame:
        """
        ✅ CORRECCIÓN V12: Carga historial completo desde BD.
        La historia es lo más importante para entender el comportamiento completo.
        """
        try:
            with self.db.engine.connect() as conn:
                query = "SELECT * FROM normalized_data_table ORDER BY timestamp"
                historical_data = pd.read_sql_query(query, conn)
                
                if historical_data.empty:
                    logger.warning("No hay datos históricos en la BD. Retornando DataFrame vacío.")
                    return pd.DataFrame()
                
                # Convertir timestamp a datetime si existe
                date_col = next((c for c in ['timestamp', 'fecha', 'datetime', 'date'] 
                                if c in historical_data.columns), None)
                if date_col:
                    historical_data[date_col] = pd.to_datetime(historical_data[date_col], errors='coerce')
                
                logger.info(f"✅ Historial completo cargado: {len(historical_data)} registros")
                return historical_data
                
        except Exception as e:
            logger.error(f"Error cargando historial completo: {e}", exc_info=True)
            return pd.DataFrame()

    def run_incremental_pipeline(self, file_path: str):
        """
        ✅ CORRECCIÓN V12: Orquestación por Lotes con análisis sobre historial completo.
        
        Flujo corregido:
        1. Agregar nuevos datos a BD
        2. Cargar historial COMPLETO desde BD
        3. Seleccionar, modelar y predecir sobre historial COMPLETO
        4. Aplicar ponderación 70/30 solo en estadísticas (no en modelos)
        """
        try:
            logger.info(f"=== Iniciando Pipeline Incremental (Realidad Industrial) ===")
            
            # 1. Obtener Watermark para Delta Processing
            last_ts = self.db.get_last_processed_timestamp("system_global")
            logger.info(f"Procesando datos posteriores a: {last_ts}")
            
            # 2. Carga Filtrada (Simulación de ETL Delta) - Solo datos nuevos
            new_data_raw = self.preprocessor.load_data(file_path, since_timestamp=last_ts)
            
            if new_data_raw.empty:
                logger.info("No hay nuevos datos en el PLC/Fuente para procesar.")
                return None
            
            # 3. Procesamiento Robust de datos nuevos
            clean_data = self.preprocessor.clean_data(new_data_raw)
            
            # 4. ✅ Normalización con scaler pre-entrenado (fit_mode=False)
            # Usa parámetros del scaler entrenado previamente (previene data leakage)
            normalized_new_data = self.preprocessor.normalize_data(clean_data, fit_mode=False)
            
            # 5. ✅ Guardar datos nuevos en histórico (APPEND al historial)
            self.preprocessor.save_to_db(normalized_new_data, "normalized_data_table")
            logger.info(f"✅ Datos nuevos agregados al historial: {len(normalized_new_data)} registros")
            
            # 6. ✅ CORRECCIÓN V12: Cargar historial COMPLETO desde BD
            # La historia es lo más importante para entender el comportamiento completo
            full_historical_data = self._load_full_historical_data()
            
            if full_historical_data.empty:
                logger.warning("No hay historial completo disponible. Usando solo datos nuevos.")
                full_historical_data = normalized_new_data
            
            # 7. ✅ CORRECCIÓN V12: Selección SOBRE HISTORIAL COMPLETO
            # El análisis debe hacerse sobre todos los datos históricos, no solo los nuevos
            critical_variables = self.selector.select_critical_variables(full_historical_data)
            logger.info(f"✅ Variables críticas seleccionadas sobre historial completo: {len(critical_variables)}")
            
            # ✅ MEJORA V14: Detectar evolución dinámica de variables
            # Obtener categorías actuales y variables con baseline existente
            current_categories = self.selector.get_variable_categories()
            variables_with_baseline = self.modeler.get_variables_with_baseline()
            
            # ✅ MEJORA V15: Incluir variables en monitoreo para detección completa de fallas
            # Obtener variables en monitoreo desde categorías actuales
            monitoring_variables = [var for var, cat in current_categories.items() 
                                  if cat == 'monitoring']
            logger.info(f"✅ Variables en monitoreo identificadas: {len(monitoring_variables)}")
            
            # Combinar variables críticas y en monitoreo (similar a dashboard)
            variables_to_model = list(critical_variables)
            for v in monitoring_variables:
                if v not in variables_to_model:
                    variables_to_model.append(v)
            
            logger.info(f"✅ Total de variables a modelar (críticas + monitoreo): {len(variables_to_model)} "
                       f"(críticas={len(critical_variables)}, monitoreo={len([v for v in variables_to_model if v in monitoring_variables])})")
            
            # Detectar nuevas variables (críticas o monitoreo) que no tienen baseline
            new_critical_variables = [var for var in critical_variables 
                                     if not self.modeler.has_baseline(var)]
            new_monitoring_variables = [var for var in monitoring_variables 
                                       if not self.modeler.has_baseline(var)]
            new_variables = new_critical_variables + new_monitoring_variables
            
            # Detectar variables que cambiaron de categoría
            # Obtener categorías previas desde BD (última ejecución)
            previous_categories = self._get_previous_categories()
            category_changes = self._detect_category_changes(previous_categories, current_categories)
            
            # Logging detallado de evolución
            if new_critical_variables:
                logger.info(f"🆕 NUEVAS VARIABLES CRÍTICAS detectadas ({len(new_critical_variables)}): {new_critical_variables}")
                logger.info("   → Estas variables ahora requieren baseline y serán entrenadas automáticamente")
            
            if new_monitoring_variables:
                logger.info(f"🆕 NUEVAS VARIABLES EN MONITOREO detectadas ({len(new_monitoring_variables)}): {new_monitoring_variables}")
                logger.info("   → Estas variables ahora requieren baseline y serán entrenadas automáticamente")
            
            if category_changes:
                logger.info(f"🔄 CAMBIOS EN CATEGORÍAS detectados ({len(category_changes)}):")
                for var, (old_cat, new_cat) in category_changes.items():
                    logger.info(f"   → {var}: {old_cat} → {new_cat}")
                    if new_cat == 'critical' and old_cat != 'critical':
                        logger.info(f"      ⚠️  Variable ahora es CRÍTICA y requiere baseline")
                    elif new_cat == 'monitoring' and old_cat not in ['critical', 'monitoring']:
                        logger.info(f"      ⚠️  Variable ahora está en MONITOREO y requiere baseline")
            
            # 8. ✅ CORRECCIÓN V12 + MEJORA V14 + V15: Modelado SOBRE HISTORIAL COMPLETO
            # Los modelos deben entrenarse con todos los datos históricos para capturar patrones completos
            # Ahora también entrena automáticamente nuevas variables críticas Y en monitoreo
            for var in variables_to_model:
                if var in full_historical_data.columns:
                    is_new = var in new_variables
                    is_critical = var in critical_variables
                    is_monitoring = var in monitoring_variables
                    
                    # Determinar prefijo de log según tipo y si es nueva
                    if is_new:
                        if is_critical:
                            log_prefix = "🆕 ENTRENANDO NUEVA VARIABLE CRÍTICA"
                        else:
                            log_prefix = "🆕 ENTRENANDO NUEVA VARIABLE EN MONITOREO"
                    else:
                        if is_critical:
                            log_prefix = "Actualizando baseline incremental (CRÍTICA)"
                        else:
                            log_prefix = "Actualizando baseline incremental (MONITOREO)"
                    
                    logger.info(f"{log_prefix} para {var} sobre historial completo...")
                    
                    # ✅ Agnóstico temporal: usar timestamp real como índice si existe
                    if 'timestamp' in full_historical_data.columns:
                        series_for_model = full_historical_data.set_index(
                            pd.to_datetime(full_historical_data['timestamp'])
                        )[var]
                    elif 'fecha' in full_historical_data.columns:
                        series_for_model = full_historical_data.set_index(
                            pd.to_datetime(full_historical_data['fecha'])
                        )[var]
                    else:
                        series_for_model = full_historical_data[var]
                    
                    # Entrenar sobre historial completo
                    results = self.modeler.fit_ensemble(var, series_for_model)
                    
                    # ✅ save_baseline aplicará ponderación 70/30 SOLO en estadísticas (median/iqr/p95/p05)
                    # Los modelos (SARIMAX/Prophet) se re-entrenan completamente con historial completo
                    self.modeler.save_baseline(results)
                    
                    if is_new:
                        category_type = "crítica" if is_critical else "en monitoreo"
                        logger.info(f"✅ Baseline entrenado exitosamente para nueva variable {category_type}: {var}")
            
            # 9. ✅ CORRECCIÓN V12: Predicción SOBRE HISTORIAL COMPLETO
            # La probabilidad debe calcularse sobre todos los datos históricos
            prediction = self.predictor.predict(full_historical_data)
            
            # 10. ✅ Generar forecast de próximas 24 horas (nuevo en V11)
            forecast_result = self.predictor.generate_forecast(forecast_horizon_hours=24)
            prediction['forecast'] = forecast_result
            
            self._report_results(prediction)
            
            # 11. Actualizar Watermark global
            date_col = next((c for c in ['fecha', 'timestamp'] if c in clean_data.columns), None)
            if date_col:
                max_ts = clean_data[date_col].max()
                self.db.update_batch_control("system_global", max_ts)
                logger.info(f"Watermark actualizado a: {max_ts}")
                
            return prediction

        except Exception as e:
            logger.error(f"Error en el pipeline incremental: {e}", exc_info=True)

    def _get_previous_categories(self) -> Dict[str, str]:
        """
        ✅ MEJORA V14: Obtiene categorías de variables de la última ejecución desde BD.
        Retorna dict {variable_name: category} para comparar con categorías actuales.
        """
        previous_categories = {}
        try:
            with self.db.engine.connect() as conn:
                # Obtener la última ejecución de análisis (más reciente timestamp)
                result = conn.execute(text("""
                    SELECT DISTINCT ON (variable_name) 
                           variable_name, category
                    FROM variable_analysis_history
                    ORDER BY variable_name, analyzed_at DESC
                """)).fetchall()
                
                for row in result:
                    previous_categories[row[0]] = row[1]
        except Exception as e:
            logger.warning(f"Error obteniendo categorías previas: {e}. Asumiendo primera ejecución.")
        
        return previous_categories

    def _detect_category_changes(self, previous: Dict[str, str], current: Dict[str, str]) -> Dict[str, Tuple[str, str]]:
        """
        ✅ MEJORA V14: Detecta cambios en categorías de variables.
        Retorna dict {variable_name: (old_category, new_category)} para variables que cambiaron.
        """
        changes = {}
        
        # Variables que cambiaron de categoría
        for var, new_cat in current.items():
            old_cat = previous.get(var, None)
            if old_cat is not None and old_cat != new_cat:
                changes[var] = (old_cat, new_cat)
        
        # Variables que aparecieron por primera vez (no estaban en previous)
        for var, new_cat in current.items():
            if var not in previous:
                changes[var] = ('unknown', new_cat)
        
        return changes

    def _report_results(self, res):
        """Genera el reporte de estado en consola."""
        if not res: return
        print("\n" + "="*40)
        print(f"ESTADO INDUSTRIAL: {res['system_health']['status'].upper()}")
        print(f"PROBABILIDAD DE FALLA: {res['system_health']['probability']:.1%}")
        print("="*40)
        if res['top_influencers']:
            print("Variables Alarmantes:")
            for var, prob in res['top_influencers']:
                print(f"  - {var}: {prob:.1%}")
        print("="*40 + "\n")

if __name__ == "__main__":
    DATA_PATH = "filtered_consolidated_data_cleaned.xlsx"
    engine = PrognosisEngine()
    
    # En producción se llamaría a run_incremental_pipeline
    # Para el ambiente de pruebas (Notebook symmetry), usamos run_pipeline
    engine.run_pipeline(DATA_PATH)
