# Prognosis II - Industrial Health Monitoring System

**Sistema de Prognosis Industrial Adaptativa Basado en Ensemble Híbrido y Aprendizaje Incremental**

Sistema modularizado, robusto y agnóstico para mantenimiento predictivo industrial que predice probabilidades de falla mediante análisis estadístico de mediciones multivariadas.

---

## 🎯 Características Principales

- **🔍 Detección Temprana de Fallas:** Predicción de probabilidades de falla mediante análisis estadístico
- **🤖 Selección Automática de Variables:** Sistema multifactorial que identifica variables críticas automáticamente
- **📊 Ensemble Híbrido:** SARIMAX + Prophet + Isolation Forest para línea base adaptativa
- **🧠 Explainable AI (XAI):** SHAP para explicar qué variables causan el riesgo
- **🔄 Aprendizaje Incremental:** Adaptación continua mediante ponderación 70/30 y re-entrenamiento
- **🌐 Agnóstico al Activo:** Funciona para cualquier tipo de activo industrial mediante propiedades estadísticas
- **📈 Trazabilidad Completa:** Versionado de modelos, auditoría y persistencia en PostgreSQL
- **🐳 Docker Ready:** Despliegue rápido con docker-compose

---

## 🚀 Inicio Rápido

### Desarrollo Local

1. **Clonar el repositorio:**
   ```bash
   git clone <repository-url>
   cd Prognosis_II
   ```

2. **Crear entorno virtual:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # Linux/Mac
   # o
   venv\Scripts\activate  # Windows
   ```

3. **Instalar dependencias:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configurar base de datos:**
   - Asegurar que PostgreSQL esté corriendo
   - Configurar variables de entorno (opcional, ver `.env.example`)

5. **Ejecutar Dashboard:**
   ```bash
   streamlit run src/ui/dashboard.py
   ```

6. **Acceder al dashboard:**
   - Abrir navegador en `http://localhost:8501`

### Despliegue con Docker

1. **Construir y ejecutar con docker-compose:**
   ```bash
   docker-compose up -d
   ```

2. **Acceder al dashboard:**
   - Abrir navegador en `http://localhost:8501`

3. **Ver logs:**
   ```bash
   docker-compose logs -f app
   ```

---

## 📚 Documentación

### Documentación Técnica Completa

**📖 [DOCUMENTACION_TECNICA_COMPLETA.md](DOCUMENTACION_TECNICA_COMPLETA.md)**

Documentación exhaustiva que incluye:
- Arquitectura del sistema (capas, componentes, diagramas)
- Componentes y módulos (responsabilidades, métodos, dependencias)
- Flujos de datos (inicialización, incremental, forecast)
- Modelos y algoritmos (SARIMAX, Prophet, Isolation Forest, fórmulas)
- Base de datos (esquema, tablas, transacciones)
- API e interfaces (dashboard, programática)
- Configuración (parámetros, variables de entorno)
- Despliegue (Docker, docker-compose, producción)
- Operación y monitoreo (logs, métricas)

### Estado del Arte

**📖 [ESTADO_DEL_ARTE_PROGNOSIS_INDUSTRIAL.md](ESTADO_DEL_ARTE_PROGNOSIS_INDUSTRIAL.md)**

Investigación del estado del arte con referencias científicas:
- Fundamentos de PHM (Prognostics and Health Management)
- Series temporales y forecasting (ARIMA, SARIMAX, Prophet)
- Detección de anomalías (Isolation Forest)
- Selección de variables y feature engineering
- Aprendizaje incremental y concept drift
- Explainable AI (XAI) con SHAP
- MLOps y gobernanza de modelos
- Comparación con sistemas existentes
- Contribuciones y novedades del sistema
- Referencias bibliográficas completas

### Explicación Profunda del Funcionamiento

**📖 [EXPLICACION_PROFUNDA_FUNCIONAMIENTO_APLICACION_V13.md](EXPLICACION_PROFUNDA_FUNCIONAMIENTO_APLICACION_V13.md)**

Explicación detallada de cómo funciona la aplicación:
- Diferencia entre variables críticas y probabilidad de falla
- Con qué variables se realizan los modelos
- Cómo se construye la línea base
- Cómo se calcula la probabilidad de falla (paso a paso)
- Cómo opera en producción (batches incrementales)
- Qué es el backtest y sus componentes
- Evaluación científica
- El reporte detallado

---

## 🏗️ Arquitectura del Sistema

```
┌─────────────────────────────────────────────────────────────┐
│                    CAPA DE PRESENTACIÓN                      │
│  Streamlit Dashboard (src/ui/dashboard.py)                  │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    CAPA DE APLICACIÓN                       │
│  PrognosisEngine (prognosis_engine.py)                      │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    CAPA DE LÓGICA DE NEGOCIO                │
│  DataPreprocessor │ KeyVariableSelector │ BaselineModeler  │
│  Predictor         │ XAIExplainer        │ Validator        │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    CAPA DE PERSISTENCIA                     │
│  DatabaseManager (PostgreSQL)                               │
└─────────────────────────────────────────────────────────────┘
```

### Pipeline de 4 Fases

1. **Ingesta y Acondicionamiento:** ETL robusto, normalización Unicode, estandarización Z-score
2. **Selección Multifactorial:** Score basado en varianza, estabilidad, tendencia, correlación
3. **Línea Base Adaptativa:** Ensemble híbrido (SARIMAX + Prophet + Isolation Forest)
4. **Inferencia y Alertas:** Cálculo de probabilidades, XAI, alertas categorizadas

---

## 📦 Estructura del Proyecto

```
Prognosis_II/
├── src/
│   ├── config/              # Configuración del sistema
│   │   ├── system_config.py
│   │   └── asset_config_manager.py
│   ├── data/                # Preprocesamiento y BD
│   │   ├── preprocessor.py
│   │   └── db_manager.py
│   ├── features/            # Selección de variables
│   │   └── selector.py
│   ├── models/              # Modelado y predicción
│   │   ├── baseline_modeler.py
│   │   ├── predictor.py
│   │   └── xai_explainer.py
│   ├── validation/         # Validación científica
│   │   └── scientific_validator.py
│   ├── mlops/               # MLOps y monitoreo
│   │   ├── model_registry.py
│   │   └── continuous_monitor.py
│   └── ui/                  # Dashboard
│       └── dashboard.py
├── configs/                 # Configuraciones de activos
│   └── assets/
├── tests/                   # Tests unitarios e integración
│   ├── unit/
│   └── integration/
├── prognosis_engine.py      # Motor principal
├── docker-compose.yml       # Orquestación Docker
├── Dockerfile               # Imagen Docker
├── requirements.txt         # Dependencias Python
└── README.md               # Este archivo
```

---

## 🔧 Configuración

### Variables de Entorno

Crear archivo `.env` (opcional):

```bash
DB_USER=postgres
DB_PASSWORD=industrial2024
DB_HOST=localhost
DB_PORT=5432
DB_NAME=prognosis_db
```

### Parámetros del Sistema

Ver `src/config/system_config.py` para:
- Umbrales estadísticos
- Parámetros de modelado
- Umbrales de alerta

---

## 📊 Uso del Sistema

### Modo Dashboard (Interactivo)

1. Ejecutar: `streamlit run src/ui/dashboard.py`
2. Configurar ruta de datos en sidebar
3. Ajustar parámetros (umbrales, número de variables)
4. Presionar "Ejecutar Prognosis"
5. Visualizar resultados en dashboard

### Modo Programático

```python
from prognosis_engine import PrognosisEngine

# Inicializar motor
engine = PrognosisEngine()

# Pipeline completo (primera ejecución)
prediction = engine.run_pipeline("data.xlsx")

# Pipeline incremental (batches)
prediction = engine.run_incremental_pipeline("new_data.xlsx")

# Acceder a resultados
print(f"Probabilidad de falla: {prediction['system_health']['probability']:.1%}")
print(f"Estado: {prediction['system_health']['status']}")
print(f"Alertas: {len(prediction['alerts'])}")
```

---

## 🧪 Testing

### Ejecutar Tests

```bash
# Todos los tests
python run_tests.py

# Tests unitarios específicos
python -m pytest tests/unit/

# Tests de integración
python -m pytest tests/integration/
```

---

## 📈 Métricas y Validación

### Métricas de Modelos

- **RMSE:** Root Mean Squared Error
- **MAE:** Mean Absolute Error
- **R²:** Coefficient of Determination
- **MAPE:** Mean Absolute Percentage Error
- **AIC/BIC:** Criterios de información

### Validación Científica

- **Backtest Temporal:** Validación cruzada temporal
- **Test de Diebold-Mariano:** Comparación con baseline naive
- **Detección de Concept Drift:** Monitoreo continuo

---

## 🔬 Fundamentos Científicos

El sistema se fundamenta en:

- **Box & Jenkins (1976):** ARIMA/SARIMAX para series temporales
- **Taylor & Letham (2018):** Prophet para forecasting aditivo
- **Liu et al. (2008):** Isolation Forest para detección de anomalías
- **Lundberg & Lee (2017):** SHAP para explicabilidad
- **Gama et al. (2014):** Concept drift y aprendizaje incremental

Ver [ESTADO_DEL_ARTE_PROGNOSIS_INDUSTRIAL.md](ESTADO_DEL_ARTE_PROGNOSIS_INDUSTRIAL.md) para referencias completas.

---

## 🎓 Contribuciones Científicas

1. **Selección Automática de Variables:** Sistema multifactorial agnóstico
2. **Ensemble Híbrido:** Combinación de modelos temporales y detección de anomalías
3. **Aprendizaje Incremental Híbrido:** Ponderación 70/30 en estadísticas, re-entrenamiento en modelos
4. **XAI Nativo:** SHAP integrado desde el diseño
5. **Agnosticidad Total:** Funciona para cualquier activo industrial

---

## 📝 Licencia

Proprietary - Developed for Industrial Insights

---

## 👥 Autor

**Juan David Valencia Piedrahita**

---

## 📚 Referencias Rápidas

- **Documentación Técnica:** [DOCUMENTACION_TECNICA_COMPLETA.md](DOCUMENTACION_TECNICA_COMPLETA.md)
- **Estado del Arte:** [ESTADO_DEL_ARTE_PROGNOSIS_INDUSTRIAL.md](ESTADO_DEL_ARTE_PROGNOSIS_INDUSTRIAL.md)
- **Funcionamiento Detallado:** [EXPLICACION_PROFUNDA_FUNCIONAMIENTO_APLICACION_V13.md](EXPLICACION_PROFUNDA_FUNCIONAMIENTO_APLICACION_V13.md)

---

**Versión:** 1.0  
**Última actualización:** 2026-02-09
