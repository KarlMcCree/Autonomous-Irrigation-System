// firmware/field_node/field_node.ino
//
// ESP32 field node for the Autonomous Irrigation System.
//
// Responsibilities:
//   1. Read soil moisture (capacitive sensor via ADC)
//   2. Read air temperature + humidity (DHT22)
//   3. Control pump via relay
//   4. Respond to serial commands from laptop (RealHAL)
//   5. Publish data via MQTT (optional, for logging)
//   6. Run entirely standalone if laptop disconnects (offline safety)
//
// Serial protocol (matches src/hal.py → RealHAL):
//   ┌────────────────────────────────────────────────────────┐
//   │ Laptop sends         ESP32 replies                     │
//   ├────────────────────────────────────────────────────────┤
//   │ READ_MOISTURE\n   →  MOISTURE:0.42\n                   │
//   │ READ_TEMP\n       →  TEMP:26.3\n                       │
//   │ READ_HUMID\n      →  HUMID:58.2\n                      │
//   │ PUMP_ON:5.0\n     →  PUMP:25.0\n    (5s * 5ml/s = 25)  │
//   │ PUMP_OFF\n        →  PUMP:0.0\n                        │
//   │ STATUS\n          →  STATUS:ok,uptime=123s\n           │
//   └────────────────────────────────────────────────────────┘
//
// Hardware wiring:
//   ESP32 GPIO34 (ADC)  ← Soil moisture AOUT
//   ESP32 GPIO4         ↔ DHT22 DATA (+ 10kΩ pullup to 3.3V)
//   ESP32 GPIO25        → Relay IN1 (pump control)
//   ESP32 5V + GND      → Relay VCC + GND
//   Pump: 12V supply, switched through relay COM/NO
//
// ⚠️  Power warning: pump startup current can be 1A+. Power the pump
//     from a SEPARATE 12V supply with a common ground to the ESP32.

#include <WiFi.h>
#include <PubSubClient.h>
#include "DHT.h"

// ═════════════════════════════════════════════════════════════
// CONFIGURATION — edit these before flashing
// ═════════════════════════════════════════════════════════════
// WiFi (optional — system works offline if not configured)
const char* WIFI_SSID     = "YOUR_WIFI_SSID";
const char* WIFI_PASSWORD = "YOUR_WIFI_PASSWORD";

// MQTT (optional)
const char* MQTT_BROKER   = "192.168.1.100";   // laptop IP
const int   MQTT_PORT     = 1883;
const char* MQTT_TOPIC_DATA = "irrigation/data";
const char* DEVICE_ID     = "field-node-01";

// ═════════════════════════════════════════════════════════════
// PIN DEFINITIONS
// ═════════════════════════════════════════════════════════════
#define PIN_SOIL_MOISTURE   34      // ADC input (input-only pin)
#define PIN_DHT             4       // DHT22 data
#define PIN_RELAY_PUMP      25      // Relay control
#define PIN_LED_STATUS      2       // Built-in LED

#define DHT_TYPE            DHT22

// ═════════════════════════════════════════════════════════════
// SOIL MOISTURE CALIBRATION
// ═════════════════════════════════════════════════════════════
// These are ADC values (12-bit: 0-4095) from YOUR sensor.
// CALIBRATE: take readings in dry air and in a cup of water.
#define SOIL_ADC_DRY   3200    // dry air → 0% moisture
#define SOIL_ADC_WET   1200    // submerged in water → 100% moisture

// ═════════════════════════════════════════════════════════════
// PUMP CALIBRATION
// ═════════════════════════════════════════════════════════════
// Measure: run pump for 10s, weigh dispensed water, divide by 10.
#define PUMP_ML_PER_SEC  5.0   // default: 5ml per second

// ═════════════════════════════════════════════════════════════
// SAMPLING INTERVALS
// ═════════════════════════════════════════════════════════════
#define SOIL_SAMPLE_INTERVAL_MS   500       // 2 Hz median filter
#define PUBLISH_INTERVAL_MS       300000    // 5 min
#define WIFI_RETRY_INTERVAL_MS    30000     // 30 s

// ═════════════════════════════════════════════════════════════
// GLOBAL STATE
// ═════════════════════════════════════════════════════════════
DHT dht(PIN_DHT, DHT_TYPE);
WiFiClient wifiClient;
PubSubClient mqttClient(wifiClient);

unsigned long lastPublish = 0;
unsigned long lastWiFiRetry = 0;
unsigned long bootTime = 0;
float totalWaterMl = 0.0;
bool pumpActive = false;

// ═════════════════════════════════════════════════════════════
// SENSOR READING
// ═════════════════════════════════════════════════════════════
/**
 * Read soil moisture as fraction [0, 1].
 *
 * Why median filter: capacitive sensors are electrically noisy. A single
 * ADC read can fluctuate ±200 counts. We take 5 readings and pick the
 * median — rejects outliers from switching noise.
 */
float readSoilMoisture() {
    int readings[5];
    for (int i = 0; i < 5; i++) {
        readings[i] = analogRead(PIN_SOIL_MOISTURE);
        delay(10);
    }

    // Sort to find median
    for (int i = 0; i < 4; i++) {
        for (int j = i + 1; j < 5; j++) {
            if (readings[j] < readings[i]) {
                int tmp = readings[i];
                readings[i] = readings[j];
                readings[j] = tmp;
            }
        }
    }
    int raw = readings[2];

    // Convert ADC to moisture fraction
    // Higher ADC = drier soil, so invert the scale
    float moisture = (float)(SOIL_ADC_DRY - raw) /
                     (float)(SOIL_ADC_DRY - SOIL_ADC_WET);
    return constrain(moisture, 0.0, 1.0);
}

float readTemperature() {
    float t = dht.readTemperature();
    if (isnan(t)) return 25.0;  // sensible default on failure
    return t;
}

float readHumidity() {
    float h = dht.readHumidity();
    if (isnan(h)) return 50.0;
    return h;
}

// ═════════════════════════════════════════════════════════════
// PUMP CONTROL
// ═════════════════════════════════════════════════════════════
/**
 * Activate pump for `durationMs` milliseconds.
 * Returns volume dispensed in ml.
 */
float pumpOn(float durationSec) {
    unsigned long durationMs = (unsigned long)(durationSec * 1000.0);

    digitalWrite(PIN_RELAY_PUMP, HIGH);  // active-high relay
    pumpActive = true;
    Serial.printf("PUMP_START:%.1f\n", durationSec);

    unsigned long start = millis();
    while (millis() - start < durationMs) {
        delay(100);
        // Keep MQTT alive during watering
        if (mqttClient.connected()) mqttClient.loop();
    }

    digitalWrite(PIN_RELAY_PUMP, LOW);
    pumpActive = false;

    float ml = durationSec * PUMP_ML_PER_SEC;
    totalWaterMl += ml;
    return ml;
}

void pumpOff() {
    digitalWrite(PIN_RELAY_PUMP, LOW);
    pumpActive = false;
}

// ═════════════════════════════════════════════════════════════
// WIFI + MQTT (graceful — never blocks the control loop)
// ═════════════════════════════════════════════════════════════
bool connectWiFi() {
    if (WiFi.status() == WL_CONNECTED) return true;

    Serial.printf("WIFI:connecting to %s\n", WIFI_SSID);
    WiFi.mode(WIFI_STA);
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

    unsigned long start = millis();
    while (WiFi.status() != WL_CONNECTED && millis() - start < 10000) {
        delay(500);
        Serial.print(".");
    }
    Serial.println();

    if (WiFi.status() == WL_CONNECTED) {
        Serial.printf("WIFI:connected %s\n", WiFi.localIP().toString().c_str());
        return true;
    }
    Serial.println("WIFI:failed (system continues offline)");
    return false;
}

bool connectMQTT() {
    if (WiFi.status() != WL_CONNECTED) return false;
    if (mqttClient.connected()) return true;

    Serial.printf("MQTT:connecting to %s\n", MQTT_BROKER);
    if (mqttClient.connect(DEVICE_ID)) {
        Serial.println("MQTT:connected");
        return true;
    }
    Serial.printf("MQTT:failed rc=%d\n", mqttClient.state());
    return false;
}

void publishReading(float soil, float temp, float humid) {
    if (!mqttClient.connected()) return;

    char payload[256];
    snprintf(payload, sizeof(payload),
        "{\"device\":\"%s\",\"uptime\":%lu,"
        "\"soil\":%.3f,\"temp\":%.2f,\"humidity\":%.2f,"
        "\"total_ml\":%.1f,\"pump\":%s}",
        DEVICE_ID, (millis() - bootTime) / 1000,
        soil, temp, humid, totalWaterMl,
        pumpActive ? "true" : "false");

    mqttClient.publish(MQTT_TOPIC_DATA, payload);
    Serial.printf("MQTT_PUB:%s\n", payload);
}

// ═════════════════════════════════════════════════════════════
// SERIAL COMMAND HANDLER
// ═════════════════════════════════════════════════════════════
/**
 * Process one line from the laptop. Matches src/hal.py RealHAL.
 * Always sends a single-line reply ending in \n.
 */
void handleCommand(String cmd) {
    cmd.trim();
    if (cmd.length() == 0) return;

    // ── READ_MOISTURE ──
    if (cmd == "READ_MOISTURE") {
        float soil = readSoilMoisture();
        Serial.printf("MOISTURE:%.3f\n", soil);
        return;
    }

    // ── READ_TEMP ──
    if (cmd == "READ_TEMP") {
        float t = readTemperature();
        Serial.printf("TEMP:%.2f\n", t);
        return;
    }

    // ── READ_HUMID ──
    if (cmd == "READ_HUMID") {
        float h = readHumidity();
        Serial.printf("HUMID:%.2f\n", h);
        return;
    }

    // ── PUMP_ON:<seconds> ──
    if (cmd.startsWith("PUMP_ON:")) {
        float duration = cmd.substring(8).toFloat();
        if (duration <= 0 || duration > 300) {   // sanity cap 5 min
            Serial.println("PUMP:0.0\n");
            return;
        }
        float ml = pumpOn(duration);
        Serial.printf("PUMP:%.1f\n", ml);
        return;
    }

    // ── PUMP_OFF ──
    if (cmd == "PUMP_OFF") {
        pumpOff();
        Serial.println("PUMP:0.0");
        return;
    }

    // ── STATUS ──
    if (cmd == "STATUS") {
        float soil = readSoilMoisture();
        float t = readTemperature();
        float h = readHumidity();
        Serial.printf("STATUS:ok,uptime=%lus,soil=%.3f,temp=%.2f,"
                      "humid=%.2f,total_ml=%.1f,mqtt=%s\n",
                      (millis() - bootTime) / 1000,
                      soil, t, h, totalWaterMl,
                      mqttClient.connected() ? "up" : "down");
        return;
    }

    // ── CALIBRATE (helper for first-time setup) ──
    if (cmd == "CALIBRATE") {
        // Take 10 raw readings so user can see actual ADC values
        Serial.print("CALIBRATE:");
        for (int i = 0; i < 10; i++) {
            Serial.printf("%d", analogRead(PIN_SOIL_MOISTURE));
            if (i < 9) Serial.print(",");
            delay(50);
        }
        Serial.println();
        return;
    }

    // ── Unknown ──
    Serial.printf("ERROR:unknown_cmd:%s\n", cmd.c_str());
}

// ═════════════════════════════════════════════════════════════
// SETUP
// ═════════════════════════════════════════════════════════════
void setup() {
    Serial.begin(115200);
    delay(1000);   // let USB serial settle

    bootTime = millis();
    Serial.println("=== FIELD NODE BOOT ===");
    Serial.printf("Device: %s\n", DEVICE_ID);

    // GPIO setup
    pinMode(PIN_RELAY_PUMP, OUTPUT);
    digitalWrite(PIN_RELAY_PUMP, LOW);   // pump OFF at boot
    pinMode(PIN_LED_STATUS, OUTPUT);
    digitalWrite(PIN_LED_STATUS, LOW);

    // ADC resolution — 12-bit is default on ESP32, but be explicit
    analogReadResolution(12);
    analogSetPinAttenuation(PIN_SOIL_MOISTURE, ADC_11db);  // 0-3.3V range

    // DHT22
    dht.begin();

    // Take one warm-up reading (DHT22 needs ~2s after power)
    delay(2000);
    readTemperature();
    readHumidity();

    // WiFi + MQTT (best-effort)
    if (connectWiFi()) {
        mqttClient.setServer(MQTT_BROKER, MQTT_PORT);
        connectMQTT();
    }

    Serial.println("=== READY ===");
    digitalWrite(PIN_LED_STATUS, HIGH);   // LED on = alive
}

// ═════════════════════════════════════════════════════════════
// LOOP
// ═════════════════════════════════════════════════════════════
void loop() {
    // 1. Handle serial commands (highest priority — laptop is in control)
    if (Serial.available()) {
        String cmd = Serial.readStringUntil('\n');
        handleCommand(cmd);
    }

    // 2. Keep MQTT alive
    if (mqttClient.connected()) {
        mqttClient.loop();
    }

    // 3. Reconnect WiFi / MQTT if disconnected
    unsigned long now = millis();
    if (now - lastWiFiRetry > WIFI_RETRY_INTERVAL_MS) {
        lastWiFiRetry = now;
        if (WiFi.status() != WL_CONNECTED) {
            connectWiFi();
        } else if (!mqttClient.connected()) {
            connectMQTT();
        }
    }

    // 4. Periodic publish (only if connected)
    if (now - lastPublish > PUBLISH_INTERVAL_MS) {
        lastPublish = now;
        if (mqttClient.connected()) {
            float soil = readSoilMoisture();
            float t = readTemperature();
            float h = readHumidity();
            publishReading(soil, t, h);
        }
    }

    // 5. Heartbeat LED blink (1 Hz while idle)
    static unsigned long lastBlink = 0;
    static bool ledState = false;
    if (now - lastBlink > 500) {
        lastBlink = now;
        ledState = !ledState;
        digitalWrite(PIN_LED_STATUS, ledState ? HIGH : LOW);
    }

    delay(10);   // yield to WiFi/system tasks
}