import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys
import os
from pathlib import Path

# Inyectar ruta raíz para resolver importaciones industriales
root_path = str(Path(__file__).resolve().parents[2])
if root_path not in sys.path:
    sys.path.append(root_path)

from src.config import SystemConfig
from src.data.preprocessor import DataPreprocessor
from src.features.selector import KeyVariableSelector
from src.models.baseline_modeler import BaselineModeler
from src.models.predictor import IndustrialFailurePredictor
from src.data.db_manager import DatabaseManager
import os
from datetime import datetime
from typing import Dict
from typing import Any

# Estilo Premium
st.set_page_config(page_title="Prognosis II Dashboard", layout="wide", page_icon="📈")

st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stMetric {
        background-color: #1f2937;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #374151;
    }
    </style>
    """, unsafe_allow_html=True)

# Inicialización de Core
@st.cache_resource
def init_engine():
    config = SystemConfig()
    db = DatabaseManager(config)
    # Asegurar esquema MLOps
    db.ensure_database_exists()
    db.setup_mlops_schema()
    
    preprocessor = DataPreprocessor(db)
    selector = KeyVariableSelector(config, db)
    modeler = BaselineModeler(config, db)
    return config, db, preprocessor, selector, modeler

config, db, preprocessor, selector, modeler = init_engine()

if getattr(config, "USE_SQLITE", False):
    st.sidebar.info("📁 Modo SQLite local (sin PostgreSQL)")

# --- Sidebar ---
st.sidebar.title("🛠️ Configuración")
file_path = st.sidebar.text_input("Ruta Excel/CSV de Datos", "filtered_consolidated_data_cleaned.xlsx")
threshold = st.sidebar.slider("Umbral de Alerta (%)", 0, 100, int(config.ALERT_THRESHOLDS['warning']*100)) / 100
max_vars_to_model = st.sidebar.slider("Máx. variables a modelar (baseline)", 3, 30, 15)
include_monitoring = st.sidebar.checkbox("Incluir variables en observación (monitoring)", value=True)

# Aplicar umbral WARNING desde UI (para alertas/reporte)
config.ALERT_THRESHOLDS['warning'] = float(threshold)

if st.sidebar.button("🚀 Ejecutar Prognosis"):
    if os.path.exists(file_path):
        with st.spinner("Analizando señales industriales (Simetría Diamante)..."):
            # 1. Cargar y Preprocesar
            raw_data = preprocessor.load_data(file_path)
            if raw_data.empty:
                st.error("El archivo está vacío o no se pudo leer correctamente.")
                st.stop()
            
            clean_data = preprocessor.clean_data(raw_data)
            if clean_data.empty:
                st.error("No quedaron datos válidos después de la limpieza. Verifique el formato del archivo.")
                st.stop()
                
            # 2. ✅ Normalización PRIMERO (fit_mode=True para entrenamiento inicial)
            # 3. Normalización y Persistencia (Manejo de Cold Start)
            try:
                # Intentar normalizar con parámetros existentes (Producción)
                normalized_data = preprocessor.normalize_data(clean_data, fit_mode=False)
            except RuntimeError:
                # Si falla (Cold Start), entrenar scaler con el lote actual
                st.warning("⚠️ Primer arranque detectado: Calibrando Scaler Z-Score con datos actuales...")
                normalized_data = preprocessor.normalize_data(clean_data, fit_mode=True)
                
            preprocessor.save_to_db(normalized_data, "normalized_data_table")
            
            # 3. ✅ Selección de Variables SOBRE DATOS NORMALIZADOS (como notebook)
            critical_vars = selector.select_critical_variables(normalized_data)

            # Extraer categorías del selector (si están disponibles)
            selector_scores = selector.results.get('variables', {})
            monitoring_vars = [v for v, d in selector_scores.items() if d.get('category') == 'monitoring']
            discarded_vars = [v for v, d in selector_scores.items() if d.get('category') == 'discarded']
            
            # Guardar resultados del selector para el dashboard
            st.session_state['selector_results'] = selector.results
            
            if not critical_vars:
                st.warning("No se identificaron variables con anomalías claras. Usando Top 5 de mayor varianza para el diagnóstico.")
                # Selección manual si el motor es muy estricto
                numeric_cols = normalized_data.select_dtypes(include=[np.number]).columns
                # Excluir timestamp/fecha
                numeric_cols = [c for c in numeric_cols if c not in ['timestamp', 'fecha', 'date', 'datetime']]
                variances = normalized_data[numeric_cols].var().sort_values(ascending=False)
                critical_vars = variances.head(5).index.tolist()
                monitoring_vars = []
                discarded_vars = []
            
            # Variables a modelar (paridad con notebook: incluir también observación si se solicita)
            variables_to_model = list(critical_vars)
            if include_monitoring:
                # mantener orden determinista y no duplicar
                for v in monitoring_vars:
                    if v not in variables_to_model:
                        variables_to_model.append(v)

            # LIMITACIÓN INDUSTRIAL: no modelar más de N variables en UI
            if len(variables_to_model) > max_vars_to_model:
                st.info(
                    f"Se detectaron {len(critical_vars)} críticas y {len(monitoring_vars)} en observación. "
                    f"Optimizando visualización/modelado a {max_vars_to_model} variables."
                )
                variables_to_model = variables_to_model[:max_vars_to_model]
                
            if not critical_vars:
                st.error("No hay columnas numéricas disponibles para procesar.")
                st.stop()
            
            # 4. Modelado (Pipeline Granular con Feedback Visual)
            st.info(
                f"Iniciando modelado baseline de {len(variables_to_model)} variables "
                f"(críticas={len(critical_vars)}, observación={len([v for v in variables_to_model if v in monitoring_vars])})..."
            )
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Usar timestamp como índice si existe para mejorar Prophet/SARIMAX (agnóstico al activo)
            if 'timestamp' in normalized_data.columns:
                normalized_indexed = normalized_data.set_index(pd.to_datetime(normalized_data['timestamp']))
            elif 'fecha' in normalized_data.columns:
                normalized_indexed = normalized_data.set_index(pd.to_datetime(normalized_data['fecha']))
            else:
                normalized_indexed = normalized_data

            for i, var in enumerate(variables_to_model):
                status_text.text(f"🚀 Procesando {var} ({i+1}/{len(variables_to_model)})...")
                # El motor ya maneja la lógica de saltar modelos pesados si hay pocos datos
                series_for_model = normalized_indexed[var] if var in normalized_indexed.columns else normalized_data[var]
                results = modeler.fit_ensemble(var, series_for_model)
                modeler.save_baseline(results)
                progress_bar.progress((i + 1) / len(variables_to_model))
            
            status_text.empty()
            progress_bar.empty()
                
            # 5. Análisis de Probabilidad sobre TODOS los datos históricos
            # ✅ CORRECCIÓN V11: El análisis debe hacerse sobre TODOS los datos históricos
            # La historia es lo más importante para entender el comportamiento completo
            st.info(f"📊 Calculando probabilidad sobre todos los datos históricos: {len(normalized_data)} registros")
            
            predictor = IndustrialFailurePredictor(config, modeler)
            prediction = predictor.predict(normalized_data)  # ✅ TODOS los datos históricos
            
            # 6. Generar Forecast de próximas 24 horas (nuevo)
            # ✅ MEJORA V11: Forecast futuro usando modelos entrenados
            forecast_horizon_hours = 24
            forecast_result = predictor.generate_forecast(forecast_horizon_hours=forecast_horizon_hours)
            prediction['forecast'] = forecast_result  # Agregar forecast a la predicción
            
            st.session_state['results'] = {
                'timestamp': prediction.get('timestamp', datetime.now().isoformat()),
                'health': prediction['system_health'],
                'influencers': prediction['top_influencers'], # Lista de tuples (var, risk, explanation)
                'shap_explanations': prediction.get('shap_explanations', []),  # ✅ Explicaciones SHAP
                'variable_ttf': prediction.get('variable_ttf', {}),  # ✅ TTF por variable
                'alerts': prediction.get('alerts', []),  # ✅ Alertas con TTF
                'variable_risks': prediction.get('variable_risks', {}),  # ✅ Probabilidad por variable (paridad notebook)
                'data': normalized_data,
                'mapping': preprocessor.quality_report['variable_mapping'],
                'critical_vars': critical_vars,  # Variables críticas identificadas (selector)
                'monitoring_vars': monitoring_vars,  # Variables en observación (selector)
                'discarded_vars': discarded_vars,  # Variables descartadas (selector)
                'variables_modeled': variables_to_model,  # Variables para las que hay baseline
                'baseline_limits': modeler.results.get('adaptive_limits', {})  # Límites adaptativos usados
            }
            st.success("Análisis completado con Rigor Extremo.")
    else:
        st.error(f"Archivo no encontrado en {file_path}")

# --- Dashboard ---
st.title("🛡️ Prognosis II - Industrial Health Monitor")

if 'results' in st.session_state:
    res = st.session_state['results']

    def _risk_level(prob: float) -> str:
        if prob >= config.ALERT_THRESHOLDS['critical']:
            return 'CRITICAL'
        if prob >= config.ALERT_THRESHOLDS['warning']:
            return 'WARNING'
        return 'NORMAL'

    def _build_failure_report() -> str:
        ts = res.get('timestamp', datetime.now().isoformat())
        sys_prob = float(res['health'].get('probability', 0.0))
        sys_status = str(res['health'].get('status', 'normal')).upper()

        lines = [
            "",
            "=== REPORTE DE PREDICCIÓN DE FALLAS ===",
            "",
            f"Fecha: {ts}",
            "",
            "Estado del Sistema:",
            f"- Probabilidad: {sys_prob:.1%}",
            f"- Estado: {sys_status}",
            "",
            "Variables (ordenadas por probabilidad):"
        ]

        var_risks = res.get('variable_risks', {}) or {}
        for var, p in sorted(var_risks.items(), key=lambda x: x[1], reverse=True):
            lines.append(f"- {var}: {float(p):.1%} ({_risk_level(float(p))})")

        alerts = res.get('alerts') or []
        if alerts:
            lines.append("")
            lines.append("Alertas Activas:")
            for a in alerts:
                lines.append(f"- {a.get('level', '')}: {a.get('message', '')} (Prob: {float(a.get('probability', 0.0)):.1%})")

        return "\n".join(lines)
    
    # --- SECCIÓN 1: CABECERA EJECUTIVA (Zero Scroll) ---
    
    # 1.A KPIs Clave
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        prob = res['health']['probability']
        st.metric("Riesgo Global", f"{prob*100:.1f}%", help="Probabilidad combinada de falla del sistema")
    with col2:
        status = res['health']['status'].upper()
        color = "🟢" if status == "NORMAL" else "🟡" if status == "WARNING" else "🔴"
        st.metric("Estado Operativo", f"{color} {status}")
    with col3:
        ttf_hours = res['health'].get('ttf_hours')
        if ttf_hours is not None:
            if ttf_hours < 24:
                ttf_display = f"{ttf_hours:.1f}h ⚠️"
            elif ttf_hours < 72:
                ttf_display = f"{ttf_hours/24:.1f} días"
            else:
                ttf_display = f"{ttf_hours/24:.0f} días"
            st.metric("Time-to-Failure Est.", ttf_display, delta_color="inverse")
        else:
            st.metric("Salud Estimada", f"{100 - prob*100:.1f}%")
    with col4:
        active_alerts = len(res.get('alerts', []))
        st.metric("Alertas Activas", active_alerts, delta="Críticas" if any(a['level']=='CRITICAL' for a in res.get('alerts', [])) else "Normal", delta_color="inverse")

    # 1.B Alertas Críticas (Solo si existen, banner colapsable pero visible por defecto si es crítico)
    alerts = res.get('alerts', [])
    critical_alerts = [a for a in alerts if a['level'] == 'CRITICAL']
    if critical_alerts:
        st.error(f"🚨 **ATENCIÓN: {len(critical_alerts)} Alertas Críticas Detectadas**")
        with st.expander("Ver Alertas Críticas", expanded=True):
            for alert in critical_alerts:
                st.markdown(f"**🔴 {alert['variable']}**: {alert['message']}")

    # 1.C Gráfico Maestro (Risk ECG)
    st.markdown("### 📉 Tendencia Global de Riesgo")
    # Lógica de gráfico (simplificada para vista ejecutiva)
    
    # Preparar datos para ECG
    granularity = st.selectbox(
        "Horizonte de tiempo:",
        ["Últimas 24 horas", "Todos los datos", "Últimos 7 días"],
        key="ecg_granularity_exec"
    )
    
    data_df = res.get('data', pd.DataFrame()).copy()
    limits_map = res.get('baseline_limits', {}) or {}
    modeled_vars = [v for v in (res.get('variables_modeled') or []) if v in data_df.columns and v in limits_map]
    
    if 'timestamp' in data_df.columns and modeled_vars:
        x_ts = pd.to_datetime(data_df['timestamp'], errors='coerce')
        order = np.argsort(x_ts.values)
        x_ts = x_ts.iloc[order]
        df_sorted = data_df.iloc[order]
        
        if granularity == "Últimas 24 horas":
             mask = x_ts >= (x_ts.max() - pd.Timedelta(hours=24))
             x_ts, df_sorted = x_ts[mask], df_sorted[mask]
        elif granularity == "Últimos 7 días":
             mask = x_ts >= (x_ts.max() - pd.Timedelta(days=7))
             x_ts, df_sorted = x_ts[mask], df_sorted[mask]

        # Calcular prob histórica (versión optimizada para display)
        probs_by_var = []
        for v in modeled_vars[:15]: # Limitado a 15 para velocidad en render
            s = pd.to_numeric(df_sorted[v], errors='coerce')
            lim = limits_map[v]
            base, up, low = float(lim.get('baseline',0)), float(lim.get('upper',0)), float(lim.get('lower',0))
            rng = max(up - low, 1e-9)
            
            # Cálculo vectorizado rápido
            dev = ((s - base) / rng).abs().clip(0,1)
            out = ((s > up) | (s < low)).astype(float)
            probs_by_var.append((0.4 * dev + 0.4 * out + 0.2 * 0.0).clip(0,1)) # Trend simplificado 0 para velocidad visual
            
        if probs_by_var:
            system_p = np.mean(np.vstack(probs_by_var), axis=0) if len(probs_by_var) > 0 else []
            fig_ecg = go.Figure()
            fig_ecg.add_trace(go.Scatter(x=x_ts, y=system_p, mode='lines', name='Riesgo Histórico', line=dict(color='#ef4444', width=2)))
            
            # Forecast (si existe)
            forecast = res.get('forecast', {})
            if forecast and 'forecast_timestamps' in forecast:
                 ft_ts = pd.to_datetime(forecast['forecast_timestamps'])
                 fig_ecg.add_trace(go.Scatter(x=ft_ts, y=[forecast['system_forecast_prob']]*len(ft_ts), mode='lines', name='Proyección', line=dict(color='#f59e0b', dash='dash')))

            fig_ecg.update_layout(template="plotly_dark", height=300, margin=dict(l=20, r=20, t=10, b=10), yaxis_range=[0,1], title_text="")
            st.plotly_chart(fig_ecg, width="stretch")
    else:
        st.info("No hay datos temporales suficientes para mostrar la tendencia.")

    # --- SECCIÓN 2: DIVULGACIÓN PROGRESIVA (Expanders) ---

    # 2.A Análisis Detallado
    with st.expander("🔍 Análisis de Causa Raíz (Variables Críticas)", expanded=False):
        c1, c2 = st.columns([1, 2])
        with c1:
            st.markdown("#### Top Influenciadores (XAI)")
            st.info(
                "**¿Qué es XAI?** (Explainable AI)\n"
                "A diferencia de una 'Caja Negra', Prognosis desglosa el riesgo en factores físicos:\n"
                "- **Desviación:** Qué tan lejos está de lo normal.\n"
                "- **Estabilidad:** Qué tan errática es la señal.\n"
                "- **Tendencia:** Qué tan rápido se deteriora."
            )
            if res['influencers']:
                influ_data = [{'Variable': i[0], 'Aporte al Riesgo': i[1]} for i in res['influencers'][:10] if len(i)>=2]
                st.dataframe(pd.DataFrame(influ_data).set_index('Variable').style.background_gradient(cmap='Reds'), width="stretch")
        
        with c2:
            st.markdown("#### Top Variables en Riesgo (Gráficas con Límites)")
            top5 = sorted(res.get('variable_risks', {}).items(), key=lambda x: x[1], reverse=True)[:3]
            
            limits_map = res.get('baseline_limits', {})
            
            for var_name, p in top5:
                if var_name in data_df.columns:
                    # Preparar datos
                    series_data = data_df[var_name].tail(100)
                    if 'timestamp' in data_df.columns:
                        timestamps = data_df['timestamp'].tail(100)
                    else:
                        timestamps = range(len(series_data))
                    
                    fig_var = go.Figure()
                    
                    # Serie Real
                    fig_var.add_trace(go.Scatter(
                        x=timestamps, y=series_data, 
                        mode='lines', name='Valor Real',
                        line=dict(color='#3b82f6', width=2)
                    ))
                    
                    # Límites Baseline
                    if var_name in limits_map:
                        lims = limits_map[var_name]
                        upper = float(lims.get('upper', 0))
                        lower = float(lims.get('lower', 0))
                        
                        fig_var.add_hline(y=upper, line_dash="dash", line_color="red", annotation_text="Límite Sup.")
                        fig_var.add_hline(y=lower, line_dash="dash", line_color="red", annotation_text="Límite Inf.")
                    
                    fig_var.update_layout(
                        title=f"{var_name} (Prob: {p:.1%})",
                        template="plotly_dark",
                        height=250,
                        margin=dict(l=20, r=20, t=30, b=20),
                        xaxis_title="Tiempo",
                        yaxis_title="Valor"
                    )
                    st.plotly_chart(fig_var, width="stretch")

    # Pre-calcular reporte para uso en descarga y visualización
    report_text = _build_failure_report()
    
    # 2.B Tabla de Datos Completa
    with st.expander("📊 Tabla de Probabilidades y Estado (Detalle TTF)", expanded=False):
        var_risks = res.get('variable_risks', {})
        if var_risks:
            # Enriquecer tabla con explicación de TTF
            st.caption(
                "**Nota sobre TTF (Time-to-Failure):** "
                "Calculado proyectando la velocidad de degradación actual ($dx/dt$) hacia el límite crítico. "
                "Una tendencia pronunciada puede reducir drásticamente el TTF aunque el valor actual esté lejos del límite."
            )
            full_df = pd.DataFrame([
                {'Variable': k, 
                 'Probabilidad': v, 
                 'Estado': _risk_level(v), 
                 'TTF (h)': res.get('variable_ttf',{}).get(k,{}).get('ttf_hours')}
                for k,v in var_risks.items()
            ]).sort_values('Probabilidad', ascending=False)
            
            # Formatear columnas
            st.dataframe(
                full_df.style.format({'Probabilidad': '{:.1%}', 'TTF (h)': '{:.1f}'})
                .background_gradient(subset=['Probabilidad'], cmap='Reds'),
                width="stretch"
            )
            
            # Descarga
            st.download_button("⬇️ Descargar Reporte TXT", report_text, "reporte_falla.txt")

    # 2.C Diagnóstico Técnico
    with st.expander("🛠️ Diagnóstico Técnico (Variables Seleccionadas)", expanded=False):
        c_diag1, c_diag2 = st.columns(2)
        with c_diag1:
            st.markdown("**Variables Críticas (Modeladas)**")
            crit_vars = res.get('critical_vars', [])
            st.dataframe(pd.DataFrame(crit_vars, columns=["Variable"]), height=200, width="stretch")
            st.caption(f"Total: {len(crit_vars)}")
            
        with c_diag2:
            st.markdown("**Variables en Observación (Monitoring)**")
            mon_vars = res.get('monitoring_vars', [])
            st.dataframe(pd.DataFrame(mon_vars, columns=["Variable"]), height=200, width="stretch")
            st.caption(f"Total: {len(mon_vars)}")

        st.divider()
        st.markdown("**Reporte Crudo del Motor (Logs)**")
        with st.container(height=300):
            st.code(report_text, language="text")

    # 9. Evaluación científica (backtest temporal) - incremental / datos reales
    with st.expander("🧪 Evaluación científica (backtest temporal)"):
        st.caption(
            "Rigor científico: para evaluar forecasting/anomalía necesitas comparar predicción vs datos futuros (holdout). "
            "Este backtest ejecuta una validación temporal simple sobre las variables top por riesgo."
        )

        horizon = st.number_input("Horizonte holdout (puntos)", min_value=6, max_value=168, value=24, step=6)
        top_k_eval = st.number_input("Número de variables a evaluar (top por riesgo)", min_value=1, max_value=10, value=3, step=1)
        run_eval = st.button("▶️ Ejecutar backtest (puede tardar)")

        def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
            y_true = np.asarray(y_true, dtype=float)
            y_pred = np.asarray(y_pred, dtype=float)
            rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
            mae = float(np.mean(np.abs(y_true - y_pred)))
            mape = float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100.0)
            return {"rmse": rmse, "mae": mae, "mape_%": mape}

        if run_eval and var_risks:
            from statsmodels.tsa.statespace.sarimax import SARIMAX
            eval_rows = []
            candidates = [v for v, _ in sorted(var_risks.items(), key=lambda x: x[1], reverse=True)[: int(top_k_eval)]]

            for v in candidates:
                if v not in res['data'].columns:
                    continue
                s = res['data'][v].dropna()
                if len(s) < int(horizon) + 20:
                    continue

                train = s.iloc[: -int(horizon)]
                test = s.iloc[-int(horizon) :]

                # 1) Naive robusto: mediana del train
                naive_pred = np.full(shape=len(test), fill_value=float(train.median()))
                naive_m = _metrics(test.values, naive_pred)

                # 2) SARIMAX (si tenemos parámetros guardados desde el ensamble)
                sarima_m = None
                sarima_metrics = None
                try:
                    v_models = modeler.models.get(v, {})
                    sarima_metrics = v_models.get("sarima_metrics", {})
                    order = sarima_metrics.get("order")
                    seasonal_order = sarima_metrics.get("seasonal_order")

                    if order is not None and seasonal_order is not None:
                        m_sar = SARIMAX(
                            train,
                            order=tuple(order),
                            seasonal_order=tuple(seasonal_order),
                            enforce_stationarity=False,
                            enforce_invertibility=False,
                        )
                        fitted = m_sar.fit(disp=False, low_memory=True, maxiter=50)
                        pred = fitted.forecast(steps=len(test))
                        sarima_m = _metrics(test.values, np.asarray(pred))
                except Exception:
                    sarima_m = None

                row = {
                    "variable": v,
                    "risk_now": float(var_risks.get(v, 0.0)),
                    "naive_rmse": naive_m["rmse"],
                    "naive_mae": naive_m["mae"],
                    "naive_mape_%": naive_m["mape_%"],
                }
                if sarima_m:
                    row.update(
                        {
                            "sarimax_rmse": sarima_m["rmse"],
                            "sarimax_mae": sarima_m["mae"],
                            "sarimax_mape_%": sarima_m["mape_%"],
                            "sarimax_aic_fullfit": (sarima_metrics or {}).get("aic"),
                        }
                    )
                eval_rows.append(row)

            if eval_rows:
                st.dataframe(pd.DataFrame(eval_rows).sort_values("risk_now", ascending=False), width="stretch")
                st.caption(
                    "Interpretación: si SARIMAX mejora RMSE/MAE vs naive, la componente de forecasting aporta valor predictivo. "
                    "Si no mejora, el riesgo se está explicando más por límites/anomalía que por forecasting."
                )
            else:
                st.warning("No se pudo ejecutar backtest (insuficientes datos por variable o faltan parámetros SARIMAX).")
else:
    st.info("👈 Configure la ruta de datos y presione 'Ejecutar Prognosis' para comenzar el diagnóstico.")

st.divider()
st.caption(f"Audit Status: Diamond Standard (Certified) | Architecture: Modular MLOps | {datetime.now().strftime('%Y-%m-%d %H:%M')}")
