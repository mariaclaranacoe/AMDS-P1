#include <WiFi.h>
#include <HTTPClient.h>
#include <SPIFFS.h>
#include <driver/i2s.h>
#include <driver/adc.h>
#include <time.h>
#include <PubSubClient.h>
#include <Preferences.h>
#include <ArduinoJson.h>
#include <algorithm> 

// WIFI + SERVER
const char* ssid = "KAYA AGRI, POULTRY & PET SUPPLY";
const char* password = "SuperB@ngadF3ynManK@ya";
const char* serverName = "http://192.168.5.200:5000/upload_audio";

// MQTT
const char* mqttServer = "192.168.5.200";
const int mqttPort = 1883;
const char* mqttClientId = "esp32_mosquito_trap_1";
const char* mqttUser = "teasis";
const char* mqttPassword = "teasis";

// Single topic for all data
const char* topic_data = "/mosquito/data";

// NTP
const long gmtOffset_sec = 8 * 3600;
const int daylightOffset_sec = 0;
const char* ntpServer = "pool.ntp.org";
bool timeSynced = false;

// SENSOR SETTINGS
const int NUM_SENSORS = 6;
const int sensorPins[NUM_SENSORS] = {36, 39, 34, 35, 32, 33};

int baseline[NUM_SENSORS] = {0};
int currentAnalog[NUM_SENSORS] = {0};

// *** MOVING AVERAGE FILTER - Reduces electrical noise ***
const int FILTER_WINDOW = 3;  // Reduced from 5 to 3 for faster response
int sensorHistory[NUM_SENSORS][3] = {{0}};
int historyIndex[NUM_SENSORS] = {0};

// *** PERSISTENCE DETECTION - Confirms real events without resetting on noise ***
int persistenceCount[NUM_SENSORS] = {0};  // Changed from hitCount
const int REQUIRED_PERSISTENCE = 3;  // Need 3 readings out of last few

// Detection settings
const float DROP_THRESHOLD_PERCENT = 10;
const int DEBOUNCE_MS = 500;
unsigned long lastTriggerTime = 0;

// *** NEW: Post-detection recovery time to prevent false second detection ***
unsigned long lastDetectionEndTime = 0;
const unsigned long POST_DETECTION_COOLDOWN = 1500;

// Calibration settings
const int BASELINE_SAMPLES = 50;
const int SENSOR_SETTLE_DELAY_MS = 10;

// Tracking variables
int detectionCount = 0;
bool isRecording = false;

// Loop timing
unsigned long lastRead = 0;
const unsigned long SENSOR_READ_INTERVAL_MS = 1;  // Changed from 2ms to 1ms (1000Hz)

unsigned long lastDisplay = 0;
const unsigned long DISPLAY_INTERVAL_MS = 500;

unsigned long lastMQTTReconnectAttempt = 0;
const unsigned long MQTT_RECONNECT_INTERVAL_MS = 5000;

// *** NEW: MQTT Heartbeat variables ***
const unsigned long MQTT_HEARTBEAT_INTERVAL_MS = 30000;
unsigned long lastMQTTHeartbeat = 0;

// DAILY RESET
const int RESET_HOUR = 7;
const int RESET_MINUTE = 0;
String lastResetDate = "";

// I2S MIC SETTINGS
#define I2S_PORT I2S_NUM_0
#define I2S_WS   16
#define I2S_SD   17
#define I2S_SCK  18

#define SAMPLE_RATE      8000
#define SAMPLE_BITS      16
#define CHANNELS         1
#define RECORD_TIME_SEC  2
#define I2S_READ_LEN     1024

const char filename[] = "/mosquito.wav";
const uint32_t WAV_DATA_SIZE = SAMPLE_RATE * (SAMPLE_BITS / 8) * CHANNELS * RECORD_TIME_SEC;

// *** NEW: BASELINE RE-LEARNING SETTINGS ***
const unsigned long BASELINE_RELEARN_INTERVAL_MS = 30 * 60 * 1000;
unsigned long lastBaselineRelearn = 0;
const int RELEARN_SAMPLES = 30;

// OBJECTS
WiFiClient espClient;
PubSubClient mqttClient(espClient);
Preferences preferences;

// FUNCTION DECLARATIONS
void connectToWiFi();
void syncTime();
String getDateTime();
String getDateOnly();
void SPIFFSInit();
void i2sInit();

void initLegacyADC();
adc1_channel_t pinToAdc1Channel(int pin);
int readSensorRaw(int pin);
int readSensorFiltered(int sensorIndex);

void calibrateSensors();
void relearnBaselines();
bool beamBroken();
void printSensorStatus();
void startRecordingAndUpload();
bool recordAudio();
bool uploadToServer();
void writeWavHeader(File &file, uint32_t dataSize);

void connectToMQTT();
void publishSensorData(const char* reason = nullptr, int oldCount = -1);
void checkDailyReset();
void resetDetectionCount();

void loadPersistentState();
void savePersistentCount();
void savePersistentResetDate();

// ADC HELPERS
adc1_channel_t pinToAdc1Channel(int pin) {
  switch (pin) {
    case 36: return ADC1_CHANNEL_0;
    case 39: return ADC1_CHANNEL_3;
    case 32: return ADC1_CHANNEL_4;
    case 33: return ADC1_CHANNEL_5;
    case 34: return ADC1_CHANNEL_6;
    case 35: return ADC1_CHANNEL_7;
    default: return ADC1_CHANNEL_0;
  }
}

void initLegacyADC() {
  adc1_config_width(ADC_WIDTH_BIT_12);
  for (int i = 0; i < NUM_SENSORS; i++) {
    adc1_config_channel_atten(pinToAdc1Channel(sensorPins[i]), ADC_ATTEN_DB_11);
  }
  Serial.println("Legacy ADC1 initialized");
}

int readSensorRaw(int pin) {
  return adc1_get_raw(pinToAdc1Channel(pin));
}

// *** MOVING AVERAGE FILTER FUNCTION ***
int readSensorFiltered(int sensorIndex) {
    int raw = readSensorRaw(sensorPins[sensorIndex]);
    
    // Store in circular buffer
    sensorHistory[sensorIndex][historyIndex[sensorIndex]] = raw;
    historyIndex[sensorIndex] = (historyIndex[sensorIndex] + 1) % FILTER_WINDOW;
    
    // Calculate average
    long sum = 0;
    for(int i = 0; i < FILTER_WINDOW; i++) {
        sum += sensorHistory[sensorIndex][i];
    }
    return sum / FILTER_WINDOW;
}

// *** NEW FUNCTION: Gently re-learn baselines to adapt to changing light ***
void relearnBaselines() {
    // Safety checks - don't re-learn if:
    // 1. We're currently recording audio
    // 2. A mosquito was just detected (within last 3 seconds)
    // 3. We're in the middle of detection cooldown
    
    if (isRecording) {
        Serial.println("Skipping baseline re-learn - recording in progress");
        return;
    }
    
    if (millis() - lastTriggerTime < 3000) {
        Serial.println("Skipping baseline re-learn - recent detection");
        return;
    }
    
    if (millis() - lastDetectionEndTime < 2000) {
        Serial.println("Skipping baseline re-learn - in cooldown period");
        return;
    }
    
    Serial.println();
    Serial.println("=== RE-LEARNING BASELINES (Adapting to light changes) ===");
    
    for (int i = 0; i < NUM_SENSORS; i++) {
        long sum = 0;
        
        // Take multiple samples for accurate reading
        for (int j = 0; j < RELEARN_SAMPLES; j++) {
            sum += readSensorFiltered(i);
            delay(10);
        }
        
        int newBaseline = sum / RELEARN_SAMPLES;
        int oldBaseline = baseline[i];
        
        // Gradual change - prevent sudden jumps (90% old, 10% new)
        baseline[i] = (oldBaseline * 9 + newBaseline) / 10;
        
        Serial.print("Sensor ");
        Serial.print(i + 1);
        Serial.print(": ");
        Serial.print(oldBaseline);
        Serial.print(" -> ");
        Serial.print(baseline[i]);
        
        int change = abs(baseline[i] - oldBaseline);
        float changePercent = 100.0 * change / oldBaseline;
        Serial.print(" (");
        Serial.print(change);
        Serial.print(" / ");
        Serial.print(changePercent, 1);
        Serial.println("%)");
        
        // Reset persistence after re-learn
        persistenceCount[i] = 0;
    }
    
    Serial.println("=== BASELINE RE-LEARN COMPLETE ===");
    Serial.println();
    
    // Publish to MQTT that baseline was updated
    if (mqttClient.connected()) {
        String msg = "{\"event\":\"baseline_update\",\"datetime\":\"" + getDateTime() + "\"}";
        mqttClient.publish(topic_data, msg.c_str(), false);
    }
}

// SETUP
void setup() {
  Serial.begin(115200);
  delay(1000);

  Serial.println();
  Serial.println("=== MOSQUITO TRAP SENSOR + AUDIO RECORDER + MQTT ===");
  Serial.println("6 break-beam sensors + INMP441 + HTTP upload");
  Serial.print("Detection threshold: ");
  Serial.print(DROP_THRESHOLD_PERCENT);
  Serial.println("% drop from baseline");
  Serial.print("Required persistence: ");
  Serial.print(REQUIRED_PERSISTENCE);
  Serial.println(" readings");
  Serial.print("Filter window: ");
  Serial.println(FILTER_WINDOW);
  Serial.print("Sensor read interval: ");
  Serial.print(SENSOR_READ_INTERVAL_MS);
  Serial.println(" ms (1000Hz sampling)");
  Serial.print("Post-detection cooldown: ");
  Serial.print(POST_DETECTION_COOLDOWN);
  Serial.println(" ms");
  Serial.print("Daily reset time: ");
  Serial.print(RESET_HOUR);
  Serial.print(":");
  if (RESET_MINUTE < 10) Serial.print("0");
  Serial.println(RESET_MINUTE);
  
  Serial.print("Baseline re-learn interval: ");
  Serial.print(BASELINE_RELEARN_INTERVAL_MS / 60000);
  Serial.println(" minutes");
  
  Serial.println("Commands: 'c' = calibrate, 'r' = reset count, 'b' = force baseline re-learn");
  Serial.println("====================================================");

  initLegacyADC();

  // Initialize sensor history buffers
  for (int i = 0; i < NUM_SENSORS; i++) {
    pinMode(sensorPins[i], INPUT);
    currentAnalog[i] = 0;
    persistenceCount[i] = 0;
    for (int j = 0; j < FILTER_WINDOW; j++) {
      sensorHistory[i][j] = 0;
    }
  }

  preferences.begin("mosqtrap", false);
  loadPersistentState();

  SPIFFSInit();
  connectToWiFi();
  syncTime();

  mqttClient.setServer(mqttServer, mqttPort);
  mqttClient.setBufferSize(512);
  connectToMQTT();

  i2sInit();
  calibrateSensors();

  lastBaselineRelearn = millis();

  if (timeSynced) {
    checkDailyReset();
  }

  publishSensorData("startup");

  Serial.println("System ready.");
  Serial.println("Press 'c' to recalibrate sensors, 'r' to reset count, 'b' to force baseline re-learn");
}

// Reset detection count function
void resetDetectionCount() {
  int oldCount = detectionCount;
  detectionCount = 0;
  savePersistentCount();
  
  Serial.println();
  Serial.println("=========================================");
  Serial.print("COUNT RESET: ");
  Serial.print(oldCount);
  Serial.println(" -> 0");
  Serial.println("=========================================");
  
  publishSensorData("manual_reset", oldCount);
}

// LOOP
void loop() {
  unsigned long now = millis();

  // Check if it's time to re-learn baselines (every 30 minutes)
  if (now - lastBaselineRelearn >= BASELINE_RELEARN_INTERVAL_MS) {
    relearnBaselines();
    lastBaselineRelearn = now;
  }

  // Improved WiFi reconnection
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("WiFi lost, reconnecting...");
    WiFi.disconnect(true);
    delay(100);
    WiFi.begin(ssid, password);
    
    int attempts = 0;
    while (WiFi.status() != WL_CONNECTED && attempts < 20) {
      delay(500);
      Serial.print(".");
      attempts++;
    }
    
    if (WiFi.status() == WL_CONNECTED) {
      Serial.println("\nWiFi reconnected!");
      Serial.print("IP: ");
      Serial.println(WiFi.localIP());
      syncTime();
    } else {
      Serial.println("\nWiFi reconnection failed");
    }
  }

  if (!mqttClient.connected()) {
    if (now - lastMQTTReconnectAttempt >= MQTT_RECONNECT_INTERVAL_MS) {
      lastMQTTReconnectAttempt = now;
      connectToMQTT();
    }
  } else {
    mqttClient.loop();
    
    if (now - lastMQTTHeartbeat >= MQTT_HEARTBEAT_INTERVAL_MS) {
      lastMQTTHeartbeat = now;
      String heartbeat = "{\"type\":\"ping\",\"ts\":\"" + getDateTime() + "\"}";
      if (mqttClient.publish(topic_data, heartbeat.c_str(), false)) {
        Serial.println("MQTT heartbeat sent");
      } else {
        Serial.println("MQTT heartbeat failed");
      }
    }
  }

  if (timeSynced) {
    checkDailyReset();
  }

  // High-speed sensor detection
  if (now - lastRead >= SENSOR_READ_INTERVAL_MS) {
    bool triggered = beamBroken();

    if (!isRecording && triggered && (now - lastTriggerTime > DEBOUNCE_MS)) {
      
      if (now - lastDetectionEndTime < POST_DETECTION_COOLDOWN) {
        lastRead = now;
        return;
      }
      
      lastTriggerTime = now;
      detectionCount++;
      savePersistentCount();

      publishSensorData("beam_break");

      isRecording = true;

      Serial.println();
      Serial.println(">>> MOSQUITO DETECTED - starting audio capture...");
      Serial.print("Time: ");
      Serial.println(getDateTime());
      Serial.print("Detection count: ");
      Serial.println(detectionCount);

      startRecordingAndUpload();

      isRecording = false;
      Serial.println("System ready again.");
      Serial.println();
    }

    lastRead = now;
  }

  if (now - lastDisplay >= DISPLAY_INTERVAL_MS) {
    printSensorStatus();
    lastDisplay = now;
  }

  // Serial command handling
  if (Serial.available()) {
    char c = Serial.read();
    if (c == 'c' || c == 'C') {
      calibrateSensors();
    } else if (c == 'r' || c == 'R') {
      resetDetectionCount();
    } 
    else if (c == 'b' || c == 'B') {
      Serial.println("Manual baseline re-learn requested...");
      relearnBaselines();
      lastBaselineRelearn = millis();
    }
  }
}

// WIFI
void connectToWiFi() {
  if (WiFi.status() == WL_CONNECTED) return;

  Serial.print("Connecting to WiFi");
  WiFi.begin(ssid, password);

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  Serial.println();
  Serial.println("WiFi connected!");
  Serial.print("ESP32 IP: ");
  Serial.println(WiFi.localIP());
}

// MQTT
void connectToMQTT() {
  if (mqttClient.connected()) return;

  Serial.print("Connecting to MQTT... ");

  String willPayload = "{";
  willPayload += "\"device\":\"esp32_trap_1\",";
  willPayload += "\"status\":\"offline\"";
  willPayload += "}";

  bool ok = mqttClient.connect(
    mqttClientId,
    mqttUser,
    mqttPassword,
    topic_data,
    1,
    true,
    willPayload.c_str()
  );

  if (ok) {
    Serial.println("connected");
    
    lastMQTTHeartbeat = millis();

    String onlinePayload = "{";
    onlinePayload += "\"device\":\"esp32_trap_1\",";
    onlinePayload += "\"status\":\"online\",";
    onlinePayload += "\"ip\":\"" + WiFi.localIP().toString() + "\",";
    onlinePayload += "\"datetime\":\"" + getDateTime() + "\"";
    onlinePayload += "}";
    mqttClient.publish(topic_data, onlinePayload.c_str(), true);
  } else {
    Serial.print("failed, rc=");
    Serial.println(mqttClient.state());
  }
}

void publishSensorData(const char* reason, int oldCount) {
  if (!mqttClient.connected()) {
    Serial.println("MQTT not connected, data not published");
    return;
  }

  StaticJsonDocument<512> jsonDoc;

  jsonDoc["datetime"] = getDateTime();
  jsonDoc["s1"] = currentAnalog[0];
  jsonDoc["s2"] = currentAnalog[1];
  jsonDoc["s3"] = currentAnalog[2];
  jsonDoc["s4"] = currentAnalog[3];
  jsonDoc["s5"] = currentAnalog[4];
  jsonDoc["s6"] = currentAnalog[5];
  jsonDoc["count"] = detectionCount;

  if (reason != nullptr) {
    jsonDoc["event"] = reason;
  }

  if (oldCount >= 0) {
    jsonDoc["old_count"] = oldCount;
  }

  char buffer[512];
  size_t n = serializeJson(jsonDoc, buffer);

  bool ok = mqttClient.publish(topic_data, (uint8_t*)buffer, n, reason == nullptr ? false : true);

  if (ok) {
    Serial.print("Published to MQTT: ");
    Serial.println(buffer);
  } else {
    Serial.println("Failed to publish to MQTT");
  }
}

// TIME
void syncTime() {
  configTime(gmtOffset_sec, daylightOffset_sec, ntpServer);
  delay(1000);

  struct tm timeinfo;
  if (getLocalTime(&timeinfo)) {
    timeSynced = true;
    Serial.print("Time synced: ");
    Serial.println(getDateTime());
  } else {
    timeSynced = false;
    Serial.println("Time sync failed. Using fallback timestamp.");
  }
}

String getDateTime() {
  struct tm timeinfo;
  if (!getLocalTime(&timeinfo)) {
    return "1970-01-01 00:00:00";
  }

  char datetime[20];
  strftime(datetime, sizeof(datetime), "%Y-%m-%d %H:%M:%S", &timeinfo);
  return String(datetime);
}

String getDateOnly() {
  struct tm timeinfo;
  if (!getLocalTime(&timeinfo)) {
    return "1970-01-01";
  }

  char datebuf[11];
  strftime(datebuf, sizeof(datebuf), "%Y-%m-%d", &timeinfo);
  return String(datebuf);
}

// DAILY RESET
void checkDailyReset() {
  struct tm timeinfo;
  if (!getLocalTime(&timeinfo)) return;

  int currentHour = timeinfo.tm_hour;
  int currentMinute = timeinfo.tm_min;
  String today = getDateOnly();

  if (lastResetDate == "") {
    lastResetDate = preferences.getString("lastResetDate", "");
    if (lastResetDate == "") {
      lastResetDate = today;
      savePersistentResetDate();
    }
  }

  bool resetTimeReached = (currentHour > RESET_HOUR) || (currentHour == RESET_HOUR && currentMinute >= RESET_MINUTE);
  bool alreadyResetToday = (lastResetDate == today);

  if (resetTimeReached && !alreadyResetToday) {
    int oldCount = detectionCount;

    Serial.println();
    Serial.println("=== DAILY RESET TRIGGERED ===");
    Serial.print("Old count: ");
    Serial.println(oldCount);

    detectionCount = 0;
    savePersistentCount();

    lastResetDate = today;
    savePersistentResetDate();

    publishSensorData("daily_reset", oldCount);

    Serial.println("Count reset to 0.");
    Serial.println("=============================");
    Serial.println();
  }
}

// PERSISTENCE
void loadPersistentState() {
  detectionCount = preferences.getInt("count", 0);
  lastResetDate = preferences.getString("lastResetDate", "");

  Serial.print("Loaded saved count: ");
  Serial.println(detectionCount);

  if (lastResetDate.length() > 0) {
    Serial.print("Last reset date: ");
    Serial.println(lastResetDate);
  } else {
    Serial.println("No saved reset date found.");
  }
}

void savePersistentCount() {
  preferences.putInt("count", detectionCount);
}

void savePersistentResetDate() {
  preferences.putString("lastResetDate", lastResetDate);
}

// SPIFFS
void SPIFFSInit() {
  if (!SPIFFS.begin(true)) {
    Serial.println("SPIFFS initialization failed!");
    while (1) {
      delay(100);
    }
  }
  Serial.println("SPIFFS initialized");
}

// I2S INIT
void i2sInit() {
    i2s_config_t i2s_config = {
        .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
        .sample_rate = SAMPLE_RATE,
        .bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT,
        .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
        .communication_format = I2S_COMM_FORMAT_I2S,
        .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
        .dma_buf_count = 8,
        .dma_buf_len = 512,
        .use_apll = true,
        .tx_desc_auto_clear = false,
        .fixed_mclk = 0
    };

    i2s_pin_config_t pin_config = {
        .bck_io_num = I2S_SCK,
        .ws_io_num = I2S_WS,
        .data_out_num = -1,
        .data_in_num = I2S_SD
    };

    i2s_driver_install(I2S_PORT, &i2s_config, 0, NULL);
    i2s_set_pin(I2S_PORT, &pin_config);
    i2s_zero_dma_buffer(I2S_PORT);
    
    Serial.println("INMP441 I2S microphone initialized");
}

// CALIBRATION FUNCTION
void calibrateSensors() {
  Serial.println();
  Serial.println("=========================================");
  Serial.println("CALIBRATING SENSORS - Keep beams CLEAR!");
  Serial.println("=========================================");
  
  for(int i = 0; i < NUM_SENSORS; i++) {
    long sum = 0;
    Serial.print("Calibrating sensor ");
    Serial.print(i + 1);
    Serial.print(" (pin ");
    Serial.print(sensorPins[i]);
    Serial.print(")... ");
    
    for(int j = 0; j < BASELINE_SAMPLES; j++) {
      int val = readSensorRaw(sensorPins[i]);
      sum += val;
      for(int k = 0; k < FILTER_WINDOW; k++) {
        sensorHistory[i][k] = val;
      }
      delay(SENSOR_SETTLE_DELAY_MS);
    }
    
    baseline[i] = sum / BASELINE_SAMPLES;
    persistenceCount[i] = 0;
    Serial.print("Baseline: ");
    Serial.println(baseline[i]);
  }
  
  Serial.println("=========================================");
  Serial.println("CALIBRATION COMPLETE!");
  Serial.print("Detection threshold: ");
  Serial.print(DROP_THRESHOLD_PERCENT);
  Serial.println("% drop");
  Serial.print("Required persistence: ");
  Serial.print(REQUIRED_PERSISTENCE);
  Serial.println(" readings");
  Serial.println("=========================================");
  Serial.println();
}

// *** FAST DETECTION FUNCTION with Persistence Counter ***
bool beamBroken() {
  bool mosquitoDetected = false;
  int detectedSensor = -1;
  float highestDrop = 0;
  
  for(int i = 0; i < NUM_SENSORS; i++) {
    // Use filtered reading for stability
    int currentValue = readSensorFiltered(i);
    currentAnalog[i] = currentValue;
    float dropPercent = 0.0;
    
    if(baseline[i] > 0) {
      dropPercent = 100.0 * (baseline[i] - currentValue) / baseline[i];
      if(dropPercent < 0) dropPercent = 0;
    }
    
    // PERSISTENCE counter - increases when beam broken, decreases slowly when not
    if(dropPercent >= DROP_THRESHOLD_PERCENT) {
      // Beam is broken, increase persistence
      persistenceCount[i]++;
      if(persistenceCount[i] > 10) persistenceCount[i] = 10;  // Cap at 10
    } else {
      // Beam is clear, decrease persistence slowly (not instantly!)
      if(persistenceCount[i] > 0) {
        persistenceCount[i]--;
      }
    }
    
    // Trigger when we have enough persistence (3 means beam broken for ~3-6ms)
    if(persistenceCount[i] >= REQUIRED_PERSISTENCE) {
      mosquitoDetected = true;
      detectedSensor = i + 1;
      if(dropPercent > highestDrop) highestDrop = dropPercent;
    }
  }
  
  if(mosquitoDetected) {
    Serial.print("Sensor ");
    Serial.print(detectedSensor);
    Serial.print(" triggered! (Drop: ");
    Serial.print(highestDrop, 1);
    Serial.println("%)");
  }
  
  return mosquitoDetected;
}

// SERIAL STATUS DISPLAY
void printSensorStatus() {
  Serial.print("[");
  Serial.print(getDateTime());
  Serial.print("] ");

  for (int i = 0; i < NUM_SENSORS; i++) {
    int val = currentAnalog[i];
    float drop = 100.0 * (baseline[i] - val) / baseline[i];
    if(drop < 0) drop = 0;

    Serial.print("S");
    Serial.print(i + 1);
    Serial.print(":");
    Serial.print(val);
    Serial.print("(");
    Serial.print(drop, 0);
    Serial.print("%)");

    // Show persistence count if active
    if(persistenceCount[i] > 0) {
      Serial.print("[");
      Serial.print(persistenceCount[i]);
      Serial.print("]");
    }

    if (i < NUM_SENSORS - 1) Serial.print(" ");
  }

  Serial.print(" | Count: ");
  Serial.print(detectionCount);

  if (isRecording) {
    Serial.print(" | RECORDING");
  }

  Serial.println();
}

// MAIN ACTION with recovery time and persistence reset
void startRecordingAndUpload() {
  if (SPIFFS.exists(filename)) {
    SPIFFS.remove(filename);
  }

  if (!recordAudio()) {
    Serial.println("Recording failed.");
    return;
  }

  if (!uploadToServer()) {
    Serial.println("Upload failed.");
    return;
  }

  Serial.println("Upload complete.");
  
  lastDetectionEndTime = millis();
  
  // Reset persistence counters after detection
  for(int i = 0; i < NUM_SENSORS; i++) {
    persistenceCount[i] = 0;
  }
  
  Serial.print("Sensor recovery period started (");
  Serial.print(POST_DETECTION_COOLDOWN);
  Serial.println("ms cooldown)");
}

// RECORD AUDIO
bool recordAudio() {
    File file = SPIFFS.open(filename, FILE_WRITE);
    if (!file) return false;
    
    writeWavHeader(file, WAV_DATA_SIZE);
    
    int16_t i2sData[I2S_READ_LEN/2];
    size_t bytesRead = 0;
    uint32_t totalBytesWritten = 0;
    
    Serial.println("Recording audio...");
    
    // Flush DMA buffer
    for (int i = 0; i < 16; i++) {
        i2s_read(I2S_PORT, i2sData, I2S_READ_LEN, &bytesRead, portMAX_DELAY);
    }
    
    while (totalBytesWritten < WAV_DATA_SIZE) {
        i2s_read(I2S_PORT, i2sData, I2S_READ_LEN, &bytesRead, portMAX_DELAY);
        
        // Apply gain
        for (size_t i = 0; i < bytesRead/2; i++) {
            int32_t amplified = (int32_t)i2sData[i] * 2;
            if (amplified > 32767) amplified = 32767;
            if (amplified < -32768) amplified = -32768;
            i2sData[i] = (int16_t)amplified;
        }
        
        size_t bytesToWrite = (bytesRead < (WAV_DATA_SIZE - totalBytesWritten)) ? bytesRead : (WAV_DATA_SIZE - totalBytesWritten);
        file.write((uint8_t*)i2sData, bytesToWrite);
        totalBytesWritten += bytesToWrite;
    }
    
    file.seek(0);
    writeWavHeader(file, totalBytesWritten);
    file.close();
    
    Serial.print("Recording finished. Bytes: ");
    Serial.println(totalBytesWritten);
    
    return true;
}

// UPLOAD TO SERVER
bool uploadToServer() {
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("WiFi not connected");
    return false;
  }

  File file = SPIFFS.open(filename, FILE_READ);
  if (!file) {
    Serial.println("Failed to open file for upload");
    return false;
  }

  HTTPClient http;
  http.begin(serverName);
  http.addHeader("Content-Type", "audio/wav");

  Serial.println("Uploading WAV file to Raspberry Pi...");

  int httpResponseCode = http.sendRequest("POST", &file, file.size());
  file.close();

  if (httpResponseCode > 0) {
    Serial.print("HTTP Response Code: ");
    Serial.println(httpResponseCode);
    String response = http.getString();
    Serial.print("Server response: ");
    Serial.println(response);
    http.end();
    return true;
  } else {
    Serial.print("Upload error: ");
    Serial.println(http.errorToString(httpResponseCode));
    http.end();
    return false;
  }
}

// WAV HEADER
void writeWavHeader(File &file, uint32_t dataSize) {
  uint8_t header[44] = {0};

  uint32_t fileSize = dataSize + 36;
  uint32_t byteRate = SAMPLE_RATE * CHANNELS * (SAMPLE_BITS / 8);
  uint16_t blockAlign = CHANNELS * (SAMPLE_BITS / 8);

  header[0]  = 'R';
  header[1]  = 'I';
  header[2]  = 'F';
  header[3]  = 'F';

  header[4]  = (uint8_t)(fileSize & 0xFF);
  header[5]  = (uint8_t)((fileSize >> 8) & 0xFF);
  header[6]  = (uint8_t)((fileSize >> 16) & 0xFF);
  header[7]  = (uint8_t)((fileSize >> 24) & 0xFF);

  header[8]  = 'W';
  header[9]  = 'A';
  header[10] = 'V';
  header[11] = 'E';

  header[12] = 'f';
  header[13] = 'm';
  header[14] = 't';
  header[15] = ' ';

  header[16] = 16;
  header[17] = 0;
  header[18] = 0;
  header[19] = 0;

  header[20] = 1;
  header[21] = 0;

  header[22] = CHANNELS;
  header[23] = 0;

  header[24] = (uint8_t)(SAMPLE_RATE & 0xFF);
  header[25] = (uint8_t)((SAMPLE_RATE >> 8) & 0xFF);
  header[26] = (uint8_t)((SAMPLE_RATE >> 16) & 0xFF);
  header[27] = (uint8_t)((SAMPLE_RATE >> 24) & 0xFF);

  header[28] = (uint8_t)(byteRate & 0xFF);
  header[29] = (uint8_t)((byteRate >> 8) & 0xFF);
  header[30] = (uint8_t)((byteRate >> 16) & 0xFF);
  header[31] = (uint8_t)((byteRate >> 24) & 0xFF);

  header[32] = (uint8_t)(blockAlign & 0xFF);
  header[33] = (uint8_t)((blockAlign >> 8) & 0xFF);

  header[34] = SAMPLE_BITS;
  header[35] = 0;

  header[36] = 'd';
  header[37] = 'a';
  header[38] = 't';
  header[39] = 'a';

  header[40] = (uint8_t)(dataSize & 0xFF);
  header[41] = (uint8_t)((dataSize >> 8) & 0xFF);
  header[42] = (uint8_t)((dataSize >> 16) & 0xFF);
  header[43] = (uint8_t)((dataSize >> 24) & 0xFF);

  file.seek(0);
  file.write(header, 44);
}
