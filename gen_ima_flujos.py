import subprocess
import os
import sys


def check_requirements():
    """Verificar que las herramientas necesarias estén instaladas"""
    try:
        # Verificar mmdc
        result = subprocess.run(
            ["mmdc", "--version"], capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            print(f"✅ mermaid-cli encontrado: {result.stdout.strip()}")
            return True
    except (
        subprocess.CalledProcessError,
        FileNotFoundError,
        subprocess.TimeoutExpired,
    ):
        pass

    try:
        # Probar con npx
        result = subprocess.run(
            ["npx", "@mermaid-js/mermaid-cli", "--version"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode == 0:
            print(f"✅ mermaid-cli encontrado via npx: {result.stdout.strip()}")
            return "npx"
    except (
        subprocess.CalledProcessError,
        FileNotFoundError,
        subprocess.TimeoutExpired,
    ):
        pass

    print("❌ mermaid-cli no encontrado")
    return False


def generate_mermaid_image(
    mermaid_code, filename, format="png", width=1920, height=1080, use_npx=False
):
    """Generar imagen desde código Mermaid con mejor manejo de errores"""

    # Crear archivo temporal
    mermaid_file = f"{filename}.mmd"
    output_file = f"{filename}.{format}"

    try:
        # Escribir código mermaid
        with open(mermaid_file, "w", encoding="utf-8") as f:
            f.write(mermaid_code)

        # Preparar comando
        if use_npx == "npx":
            cmd = ["npx", "@mermaid-js/mermaid-cli"]
        else:
            cmd = ["mmdc"]

        cmd.extend(
            [
                "-i",
                mermaid_file,
                "-o",
                output_file,
                "-w",
                str(width),
                "-H",
                str(height),
                "--backgroundColor",
                "white",
                "--theme",
                "default",
            ]
        )

        print(f"🔄 Generando {output_file}...")

        # Ejecutar comando con timeout
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

        if result.returncode == 0:
            if os.path.exists(output_file):
                file_size = os.path.getsize(output_file)
                print(f"✅ Imagen generada: {output_file} ({file_size} bytes)")
                return True
            else:
                print(f"❌ Archivo no generado: {output_file}")
                return False
        else:
            print(f"❌ Error en comando: {result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        print(f"❌ Timeout generando {filename}")
        return False
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False
    finally:
        # Limpiar archivo temporal
        if os.path.exists(mermaid_file):
            os.remove(mermaid_file)


# CÓDIGO MERMAID SIMPLIFICADO (sin emojis que pueden causar problemas)
diagram1_simple = """flowchart TD
    A["Datos Crudos<br/>7,141 registros x 57 variables<br/>Subestacion 1000kVA"] --> B
    
    subgraph F1 ["FASE 1: PREPROCESAMIENTO"]
        B["DataPreprocessor<br/>- Carga multi-formato<br/>- Normalizacion unicode<br/>- Validacion tipos"] 
        B --> C["Limpieza de Datos<br/>- Eliminacion variables constantes<br/>- Imputacion por mediana<br/>- Mapeo de nombres"]
        C --> D["Normalizacion Z-Score<br/>- StandardScaler<br/>- Media=0, Std=1<br/>- Solo variables numericas"]
        D --> E["Validacion Estructural<br/>- Timestamp obligatorio<br/>- Integridad de datos<br/>- Quality report"]
    end
    
    E --> DB[(PostgreSQL<br/>prognosis_db)]
    DB --> F
    
    subgraph F2 ["FASE 2: SELECCION DE VARIABLES"]
        F["KeyVariableSelector<br/>- Analisis multifactorial<br/>- Sistema de puntuacion<br/>- Historico de cambios"]
        F --> G["Calculo de Scores<br/>• Varianza 30%<br/>• Estabilidad 30%<br/>• Tendencia 20%<br/>• Correlacion 20%"]
        G --> H["Categorizacion<br/>• Criticas top 20%<br/>• Monitoreo top 50%<br/>• Descartadas"]
    end
    
    H --> DB2[(selected_variables_history<br/>variable_analysis_history)]
    DB --> I
    DB2 --> I
    
    subgraph F3 ["FASE 3: LINEA BASE ADAPTATIVA"]
        I["BaselineModeler<br/>- Procesamiento incremental<br/>- Ensemble de modelos<br/>- Control de versiones"]
        I --> J["Deteccion de Patrones<br/>• Tests estacionariedad<br/>• Analisis estacionalidad<br/>• Descomposicion temporal"]
        J --> K["Ensemble Models<br/>• SARIMA componentes temporales<br/>• Prophet changepoints<br/>• Isolation Forest anomalias"]
        K --> L["Validacion Temporal<br/>• Cross-validation 80/20<br/>• Metricas RMSE/MAE<br/>• Control de calidad"]
    end
    
    L --> DB3[(baseline_models<br/>pattern_registry<br/>baseline_metrics)]
    DB3 --> N
    
    subgraph F4 ["FASE 4: PREDICCION DE FALLAS"]
        N["IndustrialFailurePredictor<br/>- Comparacion vs linea base<br/>- Calculo probabilidades<br/>- Sistema de alertas"]
        N --> O["Analisis de Desviaciones<br/>• Desviacion normalizada<br/>• Valores fuera de limites<br/>• Analisis de tendencias"]
        O --> P["Calculo de Probabilidades<br/>• Factor desviacion 40%<br/>• Factor limites 40%<br/>• Factor tendencia 20%"]
        P --> Q["Generacion de Alertas<br/>• WARNING >= 70%<br/>• CRITICAL >= 90%<br/>• Estado del sistema"]
    end
    
    Q --> R["Dashboard y Resultados<br/>• Estado: NORMAL 20.3%<br/>• Variables criticas<br/>• Reportes automatizados"]
    
    classDef phaseBox fill:#e1f5fe,stroke:#01579b,stroke-width:3px
    classDef dataBox fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    classDef processBox fill:#e8f5e8,stroke:#1b5e20,stroke-width:2px
    
    class F1,F2,F3,F4 phaseBox
    class DB,DB2,DB3 dataBox
    class B,C,D,E,F,G,H,I,J,K,L,N,O,P,Q,R processBox"""

diagram2_simple = """flowchart LR
    subgraph PHASE1 ["FASE 1: ALGORITMOS DE PREPROCESAMIENTO"]
        A1["Carga Robusta<br/>• Multi-encoding fallback<br/>• Excel/CSV support<br/>• Error handling"]
        A2["Normalizacion Unicode<br/>• NFKD normalization<br/>• Diacritic removal<br/>• SQL-safe naming"]
        A3["Limpieza Inteligente<br/>• Deteccion constantes<br/>• Imputacion por mediana<br/>• Type validation"]
        A4["StandardScaler<br/>• Z-score: (X-μ)/σ<br/>• μ=0, σ=1<br/>• Preserva distribucion"]
        
        A1 --> A2 --> A3 --> A4
    end

    subgraph PHASE2 ["FASE 2: ALGORITMOS DE SELECCION"]
        B1["Score de Variabilidad<br/>score = log1p(variance) × unique_ratio<br/>• Filtro: nunique > 3<br/>• Filtro: unique_ratio > 0.001"]
        B2["Score Multi-Criterio<br/>• Varianza 30%<br/>• Estabilidad 30%<br/>• Tendencia 20%<br/>• Correlacion 20%"]
        B3["Categorizacion Adaptativa<br/>• Criticas: percentil 80<br/>• Monitoreo: percentil 50<br/>• Umbrales dinamicos"]
        
        B1 --> B2 --> B3
    end

    subgraph PHASE3 ["FASE 3: ALGORITMOS DE LINEA BASE"]
        C1["Analisis de Patrones<br/>• ADF test estacionariedad<br/>• KPSS test complementario<br/>• Seasonal decompose period=24"]
        C2["Ensemble Modeling<br/>• SARIMA(1,1,1)×(1,1,1,24)<br/>• Prophet changepoints=25<br/>• Isolation Forest contamination=0.1"]
        C3["Aprendizaje Incremental<br/>• Batches: 1000 registros<br/>• Ponderacion: 70% old, 30% new<br/>• Versioning: max 5 versiones"]
        C4["Validacion Temporal<br/>• Train/test: 80/20<br/>• RMSE + MAE metrics<br/>• Quality monitoring"]
        
        C1 --> C2 --> C3 --> C4
    end

    subgraph PHASE4 ["FASE 4: ALGORITMOS DE PREDICCION"]
        D1["Calculo de Desviaciones<br/>normalized_dev = (X - baseline) / range<br/>• Mean deviation<br/>• Max deviation<br/>• Out of limits ratio"]
        D2["Analisis de Tendencias<br/>slope, intercept = polyfit(t, values, 1)<br/>trend_strength = |slope| × n / σ"]
        D3["Probabilidad Ponderada<br/>P = 0.4×deviation + 0.4×limits + 0.2×trend<br/>• Normalizacion [0,1]<br/>• Factores combinados"]
        D4["Sistema de Alertas<br/>• WARNING: >=70%<br/>• CRITICAL: >=90%<br/>• Sistema global: mean(P_vars)"]
        
        D1 --> D2 --> D3 --> D4
    end

    PHASE1 --> PHASE2
    PHASE2 --> PHASE3  
    PHASE3 --> PHASE4

    classDef phaseStyle fill:#e3f2fd,stroke:#1976d2,stroke-width:3px
    class PHASE1,PHASE2,PHASE3,PHASE4 phaseStyle"""

# SCRIPT PRINCIPAL MEJORADO
if __name__ == "__main__":
    print("🚀 Generando diagramas de prognosis industrial...")

    # Verificar requisitos
    req_check = check_requirements()
    if not req_check:
        print("\n❌ INSTALACIÓN REQUERIDA:")
        print("1. sudo apt install nodejs npm")
        print("2. sudo npm install -g @mermaid-js/mermaid-cli")
        print("3. Reiniciar terminal y probar de nuevo")
        sys.exit(1)

    # Generar diagramas
    success_count = 0

    if generate_mermaid_image(
        diagram1_simple,
        "prognosis_pipeline_principal",
        format="png",
        width=2400,
        height=1800,
        use_npx=req_check,
    ):
        success_count += 1

    if generate_mermaid_image(
        diagram2_simple,
        "prognosis_algoritmos_detalle",
        format="png",
        width=2400,
        height=1400,
        use_npx=req_check,
    ):
        success_count += 1

    print(f"\n✅ Proceso completado: {success_count}/2 diagramas generados")

    if success_count > 0:
        print("\n📁 Archivos generados:")
        for file in os.listdir("."):
            if file.startswith("prognosis_") and file.endswith(".png"):
                size = os.path.getsize(file)
                print(f"  {file} ({size:,} bytes)")
