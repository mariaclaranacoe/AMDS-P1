#include <WiFi.h>
#include <HTTPClient.h>
#include <SPIFFS.h>
#include <driver/i2s.h>
#include <driver/adc.h>
#include <time.h>
#include <PubSubClient.h>
#include <Preferences.h>
#include <ArduinoJson.h>

// WIFI + SERVER
const char* ssid = "SBG6700AC-69159";
const char* password = "fb13122db2";
const char* serverName = "http://192.168.0.24:5000/upload_audio";

// MQTT
const char* mqttServer = "192.168.0.24";
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
int prevAnalog[NUM_SENSORS] = {0};
int currentAnalog[NUM_SENSORS] = {0};

int detectionCount = 0;
bool anyDetection = false;
bool isRecording = false;

const float REDUCTION_PERCENT = 12.0;

// Calibration / smoothing
const int BASELINE_SAMPLES = 20;
const int SENSOR_SETTLE_DELAY_MS = 30;
const int BASELINE_TRACK_LIMIT = 100;

// Trigger stability
const int REQUIRED_CONSECUTIVE_HITS = 2;
int hitCount[NUM_SENSORS] = {0};
bool sensorBlocked[NUM_SENSORS] = {false};

// Cooldown so one mosquito does not create many recordings
unsigned long lastTriggerTime = 0;
const unsigned long TRIGGER_COOLDOWN_MS = 2500;

// Loop timing
unsigned long lastRead = 0;
const unsigned long SENSOR_READ_INTERVAL_MS = 40;

unsigned long lastDisplay = 0;
const unsigned long DISPLAY_INTERVAL_MS = 2000;

unsigned long lastMQTTReconnectAttempt = 0;
const unsigned long MQTT_RECONNECT_INTERVAL_MS = 5000;

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

void calibrateSensors();
bool beamBroken();
void printSensorStatus();
void startRecordingAndUpload();
bool recordAudio();
bool uploadToServer();
void writeWavHeader(File &file, uint32_t dataSize);

void connectToMQTT();
void publishSensorData(const char* reason = nullptr, int oldCount = -1);
void checkDailyReset();

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

// SETUP
void setup() {
  Serial.begin(115200);
  delay(1000);

  Serial.println();
  Serial.println("=== MOSQUITO TRAP SENSOR + AUDIO RECORDER + MQTT ===");
  Serial.println("6 break-beam sensors + INMP441 + HTTP upload");
  Serial.print("Audio: ");
  Serial.print(SAMPLE_RATE);
  Serial.print(" Hz, ");
  Serial.print(RECORD_TIME_SEC);
  Serial.println(" sec WAV");
  Serial.print("Detection threshold: ");
  Serial.print(REDUCTION_PERCENT);
  Serial.println("% light reduction");
  Serial.print("Daily reset time: ");
  Serial.print(RESET_HOUR);
  Serial.print(":");
  if (RESET_MINUTE < 10) Serial.print("0");
  Serial.println(RESET_MINUTE);
  Serial.println("====================================================");

  initLegacyADC();

  for (int i = 0; i < NUM_SENSORS; i++) {
    pinMode(sensorPins[i], INPUT);
    prevAnalog[i] = readSensorRaw(sensorPins[i]);
    currentAnalog[i] = prevAnalog[i];
    hitCount[i] = 0;
    sensorBlocked[i] = false;
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

  if (timeSynced) {
    checkDailyReset();
  }

  // Publish startup data
  publishSensorData("startup");

  Serial.println("System ready.");
  Serial.println("Press 'c' in Serial Monitor to recalibrate sensors.");
}

// LOOP
void loop() {
  unsigned long now = millis();

  if (WiFi.status() != WL_CONNECTED) {
    connectToWiFi();
    syncTime();
  }

  if (!mqttClient.connected()) {
    if (now - lastMQTTReconnectAttempt >= MQTT_RECONNECT_INTERVAL_MS) {
      lastMQTTReconnectAttempt = now;
      connectToMQTT();
    }
  } else {
    mqttClient.loop();
  }

  if (timeSynced) {
    checkDailyReset();
  }

  if (now - lastRead >= SENSOR_READ_INTERVAL_MS) {
    bool triggered = beamBroken();

    if (!isRecording && triggered && (now - lastTriggerTime > TRIGGER_COOLDOWN_MS)) {
      lastTriggerTime = now;
      detectionCount++;
      savePersistentCount();
      
      // Publish detection data
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

  if (anyDetection || (now - lastDisplay >= DISPLAY_INTERVAL_MS)) {
    printSensorStatus();
    lastDisplay = now;
  }

  if (Serial.available()) {
    char c = Serial.read();
    if (c == 'c' || c == 'C') {
      calibrateSensors();
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
    
    // Publish online status
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

// ============= UPDATED FUNCTION TO MATCH NODE-RED EXPECTATIONS =============
void publishSensorData(const char* reason, int oldCount) {
  if (!mqttClient.connected()) {
    Serial.println("MQTT not connected, data not published");
    return;
  }

  StaticJsonDocument<512> jsonDoc;

  // Add timestamp
  jsonDoc["datetime"] = getDateTime();
  
  // Add all 6 sensor values (as numbers, not strings)
  jsonDoc["s1"] = readSensorRaw(sensorPins[0]);  // PIN 36
  jsonDoc["s2"] = readSensorRaw(sensorPins[1]);  // PIN 39
  jsonDoc["s3"] = readSensorRaw(sensorPins[2]);  // PIN 34
  jsonDoc["s4"] = readSensorRaw(sensorPins[3]);  // PIN 35
  jsonDoc["s5"] = readSensorRaw(sensorPins[4]);  // PIN 32
  jsonDoc["s6"] = readSensorRaw(sensorPins[5]);  // PIN 33
  
  // Add count (this will become total_count in Node-RED)
  jsonDoc["count"] = detectionCount;
  
  // Add event info if provided
  if (reason != nullptr) {
    jsonDoc["event"] = reason;
  }
  
  // Add reset info if provided
  if (oldCount >= 0) {
    jsonDoc["old_count"] = oldCount;
  }

  // Note: device and location are NOT included in the JSON
  // because your Node-RED function doesn't expect them

  char buffer[512];
  size_t n = serializeJson(jsonDoc, buffer);
  
  bool ok = mqttClient.publish(topic_data, (uint8_t*)buffer, n, reason == nullptr ? false : true);

  if (ok) {
    Serial.print("📡 Published to MQTT: ");
    Serial.println(buffer);
  } else {
    Serial.println("Failed to publish to MQTT");
  }
}
// ===========================================================================

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

  bool resetTimeReached =
    (currentHour > RESET_HOUR) ||
    (currentHour == RESET_HOUR && currentMinute >= RESET_MINUTE);

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

    // Publish reset event with old count
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
    .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT,
    .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
    .communication_format = I2S_COMM_FORMAT_I2S,
    .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
    .dma_buf_count = 8,
    .dma_buf_len = 256,
    .use_apll = false,
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

// SENSOR CALIBRATION
void calibrateSensors() {
  Serial.println();
  Serial.println("=== CALIBRATING SENSORS ===");
  Serial.println("Make sure all beams are CLEAR...");

  for (int i = 0; i < NUM_SENSORS; i++) {
    long sum = 0;

    for (int j = 0; j < BASELINE_SAMPLES; j++) {
      sum += readSensorRaw(sensorPins[i]);
      delay(SENSOR_SETTLE_DELAY_MS);
    }

    baseline[i] = sum / BASELINE_SAMPLES;
    prevAnalog[i] = baseline[i];
    currentAnalog[i] = baseline[i];
    hitCount[i] = 0;
    sensorBlocked[i] = false;

    Serial.print("Sensor ");
    Serial.print(i + 1);
    Serial.print(" (PIN ");
    Serial.print(sensorPins[i]);
    Serial.print(") baseline: ");
    Serial.println(baseline[i]);
  }

  Serial.println("=== CALIBRATION DONE ===");
  Serial.println();
}

// BEAM DETECTION
bool beamBroken() {
  anyDetection = false;
  bool triggered = false;

  for (int i = 0; i < NUM_SENSORS; i++) {
    int current = readSensorRaw(sensorPins[i]);
    currentAnalog[i] = current;

    float reduction = 0.0;
    if (baseline[i] > 0) {
      reduction = 100.0 * (baseline[i] - current) / baseline[i];
    }

    bool beam_drop = (reduction > REDUCTION_PERCENT);
    bool beam_clear = (reduction < (REDUCTION_PERCENT * 0.5));

    if (beam_drop) {
      hitCount[i]++;
    } else {
      hitCount[i] = 0;
    }

    if (!sensorBlocked[i] && hitCount[i] >= REQUIRED_CONSECUTIVE_HITS) {
      sensorBlocked[i] = true;
      anyDetection = true;
      triggered = true;

      Serial.print(">>> Sensor ");
      Serial.print(i + 1);
      Serial.print(" triggered | value=");
      Serial.print(current);
      Serial.print(" | baseline=");
      Serial.print(baseline[i]);
      Serial.print(" | reduction=");
      Serial.print(reduction, 1);
      Serial.println("%");
    }

    if (sensorBlocked[i] && beam_clear) {
      sensorBlocked[i] = false;
      hitCount[i] = 0;
    }

    if (!sensorBlocked[i] && abs(current - baseline[i]) < BASELINE_TRACK_LIMIT) {
      baseline[i] = (baseline[i] * 0.99) + (current * 0.01);
    }

    prevAnalog[i] = current;
  }

  return triggered;
}

// SERIAL STATUS DISPLAY
void printSensorStatus() {
  Serial.print("[");
  Serial.print(getDateTime());
  Serial.print("] ");

  for (int i = 0; i < NUM_SENSORS; i++) {
    int val = currentAnalog[i];
    int threshold = baseline[i] * (100.0 - REDUCTION_PERCENT) / 100.0;

    Serial.print("S");
    Serial.print(i + 1);
    Serial.print(":");
    Serial.print(val);

    if (sensorBlocked[i]) {
      Serial.print("X");
    } else if (val > (baseline[i] * 0.9)) {
      Serial.print("O");
    } else if (val < threshold) {
      Serial.print("v");
    } else {
      Serial.print("-");
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

// MAIN ACTION
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
}

// RECORD AUDIO
bool recordAudio() {
  File file = SPIFFS.open(filename, FILE_WRITE);
  if (!file) {
    Serial.println("Failed to open WAV file for writing");
    return false;
  }

  writeWavHeader(file, WAV_DATA_SIZE);

  uint8_t i2sData[I2S_READ_LEN];
  size_t bytesRead = 0;
  uint32_t totalBytesWritten = 0;

  Serial.println("Recording audio...");

  for (int i = 0; i < 5; i++) {
    i2s_read(I2S_PORT, i2sData, I2S_READ_LEN, &bytesRead, portMAX_DELAY);
  }

  while (totalBytesWritten < WAV_DATA_SIZE) {
    i2s_read(I2S_PORT, i2sData, I2S_READ_LEN, &bytesRead, portMAX_DELAY);

    for (size_t i = 0; i + 3 < bytesRead; i += 4) {
      int32_t sample32 =
        ((int32_t)i2sData[i + 3] << 24) |
        ((int32_t)i2sData[i + 2] << 16) |
        ((int32_t)i2sData[i + 1] << 8)  |
        ((int32_t)i2sData[i]);

      int16_t sample16 = sample32 >> 14;

      if (totalBytesWritten + 2 <= WAV_DATA_SIZE) {
        file.write((uint8_t*)&sample16, 2);
        totalBytesWritten += 2;
      }
    }
  }

  file.seek(0);
  writeWavHeader(file, totalBytesWritten);
  file.close();

  Serial.print("Recording finished. Bytes written: ");
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
