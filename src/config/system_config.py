import os

# Cargar variables de entorno (opcional)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # Si python-dotenv no está instalado, continuar sin él
    pass

class SystemConfig:
    """Configuración centralizada del sistema Prognosis II."""
    
    # Configuración de Base de Datos
    DB_USER = os.getenv("DB_USER", "postgres")
    DB_PASSWORD = os.getenv("DB_PASSWORD", "industrial2024")
    DB_HOST = os.getenv("DB_HOST", "localhost")
    DB_PORT = os.getenv("DB_PORT", "5432")
    DB_NAME = os.getenv("DB_NAME", "prognosis_db")
    
    DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    
    # Umbrales Estadísticos
    STATISTICAL_THRESHOLDS = {
        "stationarity_pvalue": 0.05,
        "seasonality_strength": 0.6,
        "anomaly_contamination": 0.05,  # Reducido del 10% para evitar sesgo
    }
    
    # Parámetros de Modelado
    MODEL_PARAMS = {
        "batch_size": 1000,
        "max_versions": 5,
        "test_size": 0.2,
        "min_data_points": 500  # Umbral para decidir si re-entrenar modelos o solo estadísticas
    }
    
    # Umbrales de Alerta
    ALERT_THRESHOLDS = {
        'warning': 0.7,
        'critical': 0.9,
        'min_deviation': 0.1
    }
