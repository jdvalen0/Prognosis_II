# Manual de Operación: Dashboard Prognosis II ("Zen Mode")

**Versión:** 1.0
**Audiencia:** Operadores, Ingenieros de Planta, Gerentes de Mantenimiento.
**Objetivo:** Guía profunda para interpretar cada indicador y gráfico del Centro de Comando.

---

## 1. Filosofía de Diseño: "Cero Scroll"
El dashboard ha sido diseñado bajo el principio de **Divulgación Progresiva**. 
*   **Nivel 1 (Siempre Visible):** Lo que necesitas saber YA (¿Está sano el sistema? ¿Cuándo va a fallar?).
*   **Nivel 2 (Desplegable):** El análisis de causa raíz.
*   **Nivel 3 (Profundo):** Tablas de datos crudos y reportes técnicos.

---

## 2. Nivel 1: Indicadores Ejecutivos (KPIs)

### 2.1. Riesgo Global (Risk Score)
*   **¿Qué es?** Un porcentaje del 0% al 100% que indica la probabilidad combinada de fallo inminente.
*   **¿Cómo se calcula?**
    Es el promedio ponderado del riesgo de todas las variables críticas, usando la fórmula de Física de Fallas:
    $$Risk 0.4 \cdot (Desviación) + 0.4 \cdot (Límite) + 0.2 \cdot (Tendencia)$$
*   **Interpretación:**
    *   **< 70% (Verde):** Operación Normal. Variaciones esperadas.
    *   **70% - 90% (Amarillo - Warning):** Estrés mecánico/térmico detectado. Planificar inspección.
    *   **> 90% (Rojo - Critical):** Falla inminente o funcional. Parada recomendada.

### 2.2. Estado Operativo
*   **¿Qué es?** Semáforo cualitativo derivado del Riesgo Global.
*   **Utilidad:** Permite una evaluación visual instantánea (menos de 1 segundo) del estado de la planta sin leer números.

### 2.3. Time-to-Failure Estimado (TTF)
*   **¿Qué es?** El tiempo restante estimado antes de que el Riesgo Global cruce el umbral crítico (90%).
*   **¿Cómo se calcula?**
    Se usa una proyección lineal basada en la pendiente de degradación actual ($dx/dt$) de las variables críticas.
    $$TTF = \frac{Umbral_{Critico} - Estado_{Actual}}{Velocidad_{Degradación}}$$
*   **Precisión:** Si el sistema es estable, dirá "Días". Si hay una falla acelerada, cambiará a "Horas" con un icono de advertencia ⚠️.

### 2.4. Tendencia Global de Riesgo (Risk ECG)
*   **¿Qué es?** El "Electrocardiograma" de la máquina. Un gráfico de línea roja continua.
*   **¿Para qué sirve?** Diferencia entre una falla súbita y un desgaste lento.
    *   *Pico repentino:* Anomalía transitoria (golpe, ruido eléctrico).
    *   *Rampa ascendente:* Falla progresiva (desgaste de rodamiento, ensuciamiento de filtro).
*   **Forecast (Línea Naranja):** Predicción de hacia dónde irá el riesgo en las próximas 24 horas.

---

## 3. Nivel 2: Diagnóstico (Expandibles)

### 3.1. 🔍 Análisis de Causa Raíz
Aquí es donde el ingeniero "profundiza".
*   **Top Influenciadores (XAI):** Un gráfico de barras que responde: *"¿Qué sensor está causando la alarma?"*.
*   **Variables Críticas:** Gráficos sparkline (miniaturas) de las 3 señales más problemáticas para ver su forma de onda reciente.

### 3.2. 📊 Tabla de Probabilidades
Acceso a la "Data Cruda".
*   Lista todas las variables ordenadas de mayor a menor riesgo.
*   Permite descargar el reporte en formato texto para enviarlo por correo o adjuntarlo a una orden de trabajo (OT).

### 3.3. 🛠️ Diagnóstico Técnico
Sección para el equipo de Data Science/IT.
*   Muestra qué variables han sido seleccionadas por el algoritmo y cuáles han sido descartadas por falta de información.
*   Ayuda a depurar si un sensor está desconectado o enviando datos planos.

---

## 4. Flujo de Trabajo Recomendado

1.  **Monitor Pasivo:** Mantener el dashboard abierto en una pantalla secundaria. Si todo está Verde/Amarillo, no requiere acción.
2.  **Alerta Activa:** Si el KPI de Estado pasa a **ROJO** o el TTF baja a horas:
    *   Abrir el expansor **🔍 Análisis de Causa Raíz**.
    *   Identificar la variable "culpable" (ej. `Temp_Rodamiento_3`).
    *   Verificar en el **Risk ECG** si es un pico o una pendiente.
3.  **Acción:** Generar reporte (Botón Descargar) y emitir orden de inspección física enfocada solo en el componente afectado.

---
**Nota:** Este dashboard no reemplaza el juicio experto, pero dirige la atención del experto hacia donde realmente importa.
