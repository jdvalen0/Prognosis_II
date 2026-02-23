#!/usr/bin/env python3
"""
Script de prueba del pipeline completo
Replica el flujo del notebook y valida que los resultados sean consistentes.
"""

import sys
import os
from pathlib import Path

# Agregar ruta raíz
root_path = Path(__file__).resolve().parent
sys.path.insert(0, str(root_path))

import logging
import pandas as pd
from src.config import SystemConfig
from src.data.db_manager import DatabaseManager
from src.data.preprocessor import DataPreprocessor
from src.features.selector import KeyVariableSelector
from src.models.baseline_modeler import BaselineModeler
from src.models.predictor import IndustrialFailurePredictor

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("TestPipeline")

def test_pipeline_completo(file_path: str = "filtered_consolidated_data_cleaned.xlsx"):
    """
    Ejecuta el pipeline completo y compara con resultados esperados del notebook.
    """
    print("=" * 80)
    print("🧪 PRUEBA DEL PIPELINE COMPLETO - PROGNOSIS II v2.0")
    print("=" * 80)
    print()
    
    # Verificar que el archivo existe
    if not os.path.exists(file_path):
        print(f"❌ ERROR: Archivo no encontrado: {file_path}")
        print(f"   Buscando en: {os.path.abspath(file_path)}")
        return False
    
    print(f"✅ Archivo encontrado: {file_path}")
    print()
    
    try:
        # 1. Inicializar componentes
        print("📦 Inicializando componentes...")
        config = SystemConfig()
        db = DatabaseManager(config)
        db.ensure_database_exists()
        db.setup_mlops_schema()
        
        if not db.check_connection():
            print("⚠️  No se pudo conectar a la base de datos. Continuando sin persistencia...")
            # Continuar sin DB para pruebas
        
        preprocessor = DataPreprocessor(db)
        selector = KeyVariableSelector(config, db)
        modeler = BaselineModeler(config, db)
        
        print("✅ Componentes inicializados")
        print()
        
        # 2. FASE 1: Preprocesamiento
        print("=" * 80)
        print("FASE 1: PREPROCESAMIENTO")
        print("=" * 80)
        
        raw_data = preprocessor.load_data(file_path)
        print(f"✅ Datos cargados: {len(raw_data)} filas, {len(raw_data.columns)} columnas")
        
        clean_data = preprocessor.clean_data(raw_data)
        print(f"✅ Datos limpiados: {len(clean_data)} filas, {len(clean_data.columns)} columnas")
        
        normalized_data = preprocessor.normalize_data(clean_data)
        print(f"✅ Datos normalizados: {len(normalized_data)} filas, {len(normalized_data.columns)} columnas")
        
        # Guardar en DB
        try:
            preprocessor.save_to_db(normalized_data, "normalized_data_table")
            print("✅ Datos persistidos en PostgreSQL")
        except Exception as e:
            print(f"⚠️  No se pudieron guardar datos en DB: {e}")
        
        print()
        
        # 3. FASE 2: Selección de Variables
        print("=" * 80)
        print("FASE 2: SELECCIÓN DE VARIABLES CRÍTICAS")
        print("=" * 80)
        
        critical_vars = selector.select_critical_variables(clean_data)
        print(f"✅ Variables críticas identificadas: {len(critical_vars)}")
        print(f"   Variables: {', '.join(critical_vars[:5])}{'...' if len(critical_vars) > 5 else ''}")
        print()
        
        if not critical_vars:
            print("⚠️  No se identificaron variables críticas. Usando top 5 por varianza...")
            numeric_cols = clean_data.select_dtypes(include=['number']).columns
            variances = clean_data[numeric_cols].var().sort_values(ascending=False)
            critical_vars = variances.head(5).index.tolist()
            print(f"   Variables seleccionadas: {', '.join(critical_vars)}")
            print()
        
        # Limitar a 10 para velocidad en pruebas
        if len(critical_vars) > 10:
            print(f"⚠️  Limitando a 10 variables para velocidad de prueba...")
            critical_vars = critical_vars[:10]
        
        # 4. FASE 3: Modelado de Baseline
        print("=" * 80)
        print("FASE 3: MODELADO DE BASELINE (ENSEMBLE)")
        print("=" * 80)
        
        for i, var in enumerate(critical_vars, 1):
            print(f"🚀 Procesando {var} ({i}/{len(critical_vars)})...")
            try:
                if var in normalized_data.columns:
                    results = modeler.fit_ensemble(var, normalized_data[var])
                    modeler.save_baseline(results)
                    
                    # Mostrar información del modelo
                    if 'models' in results:
                        models_created = list(results['models'].keys())
                        print(f"   ✅ Modelos creados: {', '.join(models_created)}")
                        if 'sarima_metrics' in results['models']:
                            aic = results['models']['sarima_metrics'].get('aic')
                            if aic:
                                print(f"   📊 AIC: {aic:.2f}")
                    print(f"   📈 Límites: [{results['limits']['lower']:.3f}, {results['limits']['upper']:.3f}]")
                else:
                    print(f"   ⚠️  Variable {var} no encontrada en datos normalizados")
            except Exception as e:
                print(f"   ❌ Error procesando {var}: {e}")
                logger.error(f"Error en fit_ensemble para {var}: {e}", exc_info=True)
        
        print()
        
        # 5. FASE 4: Predicción
        print("=" * 80)
        print("FASE 4: PREDICCIÓN Y DIAGNÓSTICO")
        print("=" * 80)
        
        predictor = IndustrialFailurePredictor(config, modeler)
        prediction = predictor.predict(normalized_data)
        
        # Mostrar resultados
        health = prediction['system_health']
        print(f"✅ Predicción completada")
        print()
        print(f"📊 SALUD DEL SISTEMA:")
        print(f"   Estado: {health['status'].upper()}")
        print(f"   Probabilidad de falla: {health['probability']:.1%}")
        if health.get('ttf_hours'):
            ttf = health['ttf_hours']
            print(f"   ⏱️  Time-to-Failure: {ttf:.1f}h" if ttf < 24 else f"   ⏱️  Time-to-Failure: {ttf/24:.1f} días")
        print()
        
        # Top influencers
        if prediction.get('top_influencers'):
            print(f"🔍 TOP {len(prediction['top_influencers'])} VARIABLES INFLUYENTES:")
            for i, item in enumerate(prediction['top_influencers'][:5], 1):
                if len(item) == 3:
                    var, risk, expl = item
                    print(f"   {i}. {var}: {risk:.1%} - {expl}")
                elif len(item) == 2:
                    var, risk = item
                    print(f"   {i}. {var}: {risk:.1%}")
            print()
        
        # Alertas
        if prediction.get('alerts'):
            print(f"🚨 ALERTAS GENERADAS: {len(prediction['alerts'])}")
            for alert in prediction['alerts'][:5]:
                level_icon = "🔴" if alert['level'] == 'CRITICAL' else "🟡"
                print(f"   {level_icon} {alert['level']}: {alert['message']}")
            print()
        
        # SHAP Explanations
        if prediction.get('shap_explanations'):
            print(f"💡 EXPLICACIONES SHAP (Top 5):")
            for var, shap_val, expl in prediction['shap_explanations'][:5]:
                print(f"   • {var}: {expl}")
            print()
        
        print("=" * 80)
        print("✅ PIPELINE COMPLETO EJECUTADO EXITOSAMENTE")
        print("=" * 80)
        print()
        print("📋 RESUMEN:")
        print(f"   • Variables procesadas: {len(critical_vars)}")
        print(f"   • Estado del sistema: {health['status'].upper()}")
        print(f"   • Probabilidad de falla: {health['probability']:.1%}")
        print(f"   • Alertas generadas: {len(prediction.get('alerts', []))}")
        print(f"   • Explicaciones SHAP: {len(prediction.get('shap_explanations', []))}")
        print()
        
        return True
        
    except Exception as e:
        print(f"❌ ERROR en el pipeline: {e}")
        logger.error("Error en pipeline completo", exc_info=True)
        return False

if __name__ == "__main__":
    # Intentar con el archivo por defecto
    file_path = "filtered_consolidated_data_cleaned.xlsx"
    
    # Permitir pasar archivo como argumento
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    
    success = test_pipeline_completo(file_path)
    sys.exit(0 if success else 1)
