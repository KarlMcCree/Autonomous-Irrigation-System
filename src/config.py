import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR      = os.path.join(PROJECT_ROOT, "data")
RAW_DIR       = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
ML_DIR        = os.path.join(PROJECT_ROOT, "ml")
DOCS_DIR      = os.path.join(PROJECT_ROOT, "docs")
LOGS_DIR      = os.path.join(PROJECT_ROOT, "logs")

# Ensure they exist (safe to call repeatedly)
for _d in (DATA_DIR, RAW_DIR, PROCESSED_DIR, ML_DIR, DOCS_DIR, LOGS_DIR):
    os.makedirs(_d, exist_ok=True)

# Data files
SOIL_CSV          = os.path.join(RAW_DIR, "soil_readings.csv")
SOIL_WINDOWS_PKL  = os.path.join(PROCESSED_DIR, "soil_windows.pkl")

# Model files — soil predictor
SOIL_MODEL_KERAS  = os.path.join(ML_DIR, "soil_predictor.keras")
SOIL_MODEL_META   = os.path.join(ML_DIR, "soil_meta.json")
SOIL_MODEL_TFLITE = os.path.join(ML_DIR, "soil_predictor.tflite")
SOIL_MODEL_H      = os.path.join(ML_DIR, "soil_predictor.h")

# Model files — plant classifier (from Project 2)
PLANT_MODEL_KERAS  = os.path.join(ML_DIR, "plant_classifier.keras")
PLANT_MODEL_META   = os.path.join(ML_DIR, "plant_meta.json")
PLANT_MODEL_TFLITE = os.path.join(ML_DIR, "plant_classifier.tflite")


HARDWARE_MODE = "simulated"

# Simulation settings

SIMULATION_CSV = os.path.join(RAW_DIR, "bhopal_agritech_dry_v2.csv")
SIMULATION_NOISE = 0.02
SIMULATION_DRIFT = 0.001

# Real hardware settings (used when HARDWARE_MODE="real")
SERIAL_PORT = "COM3"
SERIAL_BAUD = 115200

# ────────────────────────────────────────────────────────────
# HARDWARE PINS (ESP32 field node)
# ────────────────────────────────────────────────────────────
PIN_SOIL_MOISTURE  = 34        # ADC input
PIN_DHT            = 4         # DHT22 data
PIN_RELAY_PUMP     = 25        # Relay control
PIN_LED_STATUS     = 2         # Built-in LED

# ────────────────────────────────────────────────────────────
# SENSOR / SAMPLING
# ────────────────────────────────────────────────────────────
SAMPLE_INTERVAL_S    = 60      # Read sensors every 60 seconds
PUBLISH_INTERVAL_S   = 300     # Publish via MQTT every 5 min
VISION_INTERVAL_S    = 1800    # Run vision inference every 30 min

# ────────────────────────────────────────────────────────────
# SOIL MOISTURE CALIBRATION
# ────────────────────────────────────────────────────────────
SOIL_ADC_DRY = 3200            # ADC value in dry air (0% moisture)
SOIL_ADC_WET = 1200            # ADC value in water (100% moisture)

# ────────────────────────────────────────────────────────────
# PREDICTION MODEL (LSTM)
# ────────────────────────────────────────────────────────────
WINDOW_SIZE     = 30           # 30 samples = 30 min at 60s sampling
PREDICT_HORIZON = 60           # Predict 60 min ahead
N_FEATURES      = 3            # soil_moisture, temperature, humidity

LSTM_UNITS   = 32
DENSE_UNITS  = 16
DROPOUT      = 0.2

# ────────────────────────────────────────────────────────────
# TRAINING
# ────────────────────────────────────────────────────────────
BATCH_SIZE    = 32
EPOCHS        = 30
LEARNING_RATE = 1e-3
VAL_SPLIT     = 0.15
TEST_SPLIT    = 0.15
RANDOM_SEED   = 42

# ────────────────────────────────────────────────────────────
# IRRIGATION POLICIES
# ────────────────────────────────────────────────────────────
POLICY_A_THRESHOLD      = 0.30   
POLICY_B_INTERVAL_HOURS = 6      
POLICY_B_VOLUME_ML      = 100    
POLICY_C_PREDICT_THRESH = 0.25   

MAX_WATER_PER_DAY_ML    = 500    
MIN_INTERVAL_BETWEEN_ML = 90


# ────────────────────────────────────────────────────────────
# MQTT
# ────────────────────────────────────────────────────────────
MQTT_BROKER      = "localhost"
MQTT_PORT        = 1883
MQTT_TOPIC_DATA  = "irrigation/data"
MQTT_TOPIC_CMD   = "irrigation/command"
MQTT_TOPIC_ALERT = "irrigation/alert"

# ────────────────────────────────────────────────────────────
# EXPERIMENT
# ────────────────────────────────────────────────────────────
EXPERIMENT_DAYS = 7
N_PLANTS        = 3

POLICY_WATER_DURATION_S = {
    "A": 5.0,   # 25 ml per event
    "B": 15.0,  # 75 ml per event (scheduled = bigger dose, less often)
    "C": 4.0,   # 20 ml per event (predictive = smaller, more targeted)
    "D": 0.0,
}

POLICY_WATER_DURATION_S_7_DAYS = {
    "A": 10.0,   # 50 ml per event
    "B": 20.0,  # 100 ml per event (scheduled = bigger dose, less often)
    "C": 8.0,   # 40 ml per event (predictive = smaller, more targeted)
    "D": 0.0,
}