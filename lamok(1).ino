#include <WiFi.h>
#include <PubSubClient.h>
#include <ArduinoJson.h>
#include <time.h>

// WiFi and MQTT settings
const char* ssid = "SBG6700AC-69159";
const char* password = "fb13122db2";
const char* mqtt_server = "192.168.0.22";
const int mqtt_port = 1883;
const char* mqtt_user = "teasis";
const char* mqtt_password = "teasis";
const char* topic_data = "/mosquito/data";

// Sensor pins (4 sensors)
const int sensorPins[4] = {34, 35, 32, 33};
int baseline[4] = {0};
int prevAnalog[4] = {0};
int detectionCount = 0;
bool anyDetection = false;

// Detection settings
const float REDUCTION_PERCENT = 15.0;

// NTP
const long gmtOffset_sec = 8 * 3600;
const char* ntpServer = "pool.ntp.org";

WiFiClient espClient;
PubSubClient client(espClient);
bool timeSynced = false;

void setup_wifi() {
  Serial.print("Connecting to WiFi...");
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println(" Connected!");
}

String getDateTime() {
  if (!timeSynced) {
    return "1970-01-01 00:00:00";
  }
  
  struct tm timeinfo;
  if (!getLocalTime(&timeinfo)) {
    return "1970-01-01 00:00:00";
  }
  
  char datetime[20];
  strftime(datetime, sizeof(datetime), "%Y-%m-%d %H:%M:%S", &timeinfo);
  return String(datetime);
}

void reconnect() {
  while (!client.connected()) {
    Serial.print("Connecting MQTT...");
    if (client.connect("ESP32_Sensor", mqtt_user, mqtt_password)) {
      Serial.println("connected");
    } else {
      Serial.print("failed, retrying...");
      delay(5000);
    }
  }
}

void calibrateSensors() {
  Serial.println("\n=== CALIBRATING ===");
  Serial.println("Make sure beams are CLEAR...");
  
  for (int i = 0; i < 4; i++) {
    int sum = 0;
    for (int j = 0; j < 20; j++) {  // 20 readings
      sum += analogRead(sensorPins[i]);
      delay(50);
    }
    baseline[i] = sum / 20;
    
    Serial.print("Sensor ");
    Serial.print(i+1);
    Serial.print(" baseline: ");
    Serial.println(baseline[i]);
  }
  Serial.println("=== CALIBRATION DONE ===");
}

void setup() {
  Serial.begin(115200);
  delay(1000);
  
  Serial.println("\n=== MOSQUITO SENSOR ===");
  Serial.println("S1(34) S2(35) S3(32) S4(33)");
  Serial.print("Detection: ");
  Serial.print(REDUCTION_PERCENT);
  Serial.println("% light reduction");
  Serial.println("======================================");
  
  for (int i = 0; i < 4; i++) {
    pinMode(sensorPins[i], INPUT);
    prevAnalog[i] = analogRead(sensorPins[i]);
  }
  
  calibrateSensors();
  
  setup_wifi();
  
  configTime(gmtOffset_sec, 0, ntpServer);
  delay(1000);
  struct tm timeinfo;
  if (getLocalTime(&timeinfo)) {
    timeSynced = true;
    Serial.print("Time synced: ");
    Serial.println(getDateTime());
  }
  
  client.setServer(mqtt_server, mqtt_port);
}

void readSensors() {
  anyDetection = false;
  
  for (int i = 0; i < 4; i++) {
    int current = analogRead(sensorPins[i]);
    
    // Calculate percentage reduction
    float reduction = 100.0 * (baseline[i] - current) / baseline[i];
    
    // Detection
    if (reduction > REDUCTION_PERCENT && prevAnalog[i] > (baseline[i] * 0.9)) {
      detectionCount++;
      anyDetection = true;
      
      Serial.print(">>> MOSQUITO! Sensor ");
      Serial.print(i+1);
      Serial.print(" at ");
      Serial.print(getDateTime());
      Serial.print(" | Reduction: ");
      Serial.print(reduction, 1);
      Serial.print("%");
      Serial.print(" | Total: ");
      Serial.println(detectionCount);
    }
    
    // Adjust baseline slowly
    if (abs(current - baseline[i]) < 100) {
      baseline[i] = (baseline[i] * 0.99) + (current * 0.01);
    }
    
    prevAnalog[i] = current;
  }
}

void publishData() {
  StaticJsonDocument<256> jsonDoc;
  
  // ONLY THESE 3 THINGS:
  jsonDoc["s1"] = analogRead(sensorPins[0]);
  jsonDoc["s2"] = analogRead(sensorPins[1]);
  jsonDoc["s3"] = analogRead(sensorPins[2]);
  jsonDoc["s4"] = analogRead(sensorPins[3]);
  jsonDoc["count"] = detectionCount;
  jsonDoc["datetime"] = getDateTime();  // Full datetime
  
  char buffer[256];
  serializeJson(jsonDoc, buffer);
  
  if (client.publish(topic_data, buffer)) {
    Serial.print("📡 Published: ");
    Serial.println(buffer);
  }
}

void loop() {
  if (!client.connected()) reconnect();
  client.loop();
  
  static unsigned long lastRead = 0;
  if (millis() - lastRead >= 50) {
    readSensors();
    lastRead = millis();
  }
  
  static unsigned long lastDisplay = 0;
  if (anyDetection || (millis() - lastDisplay > 2000)) {
    Serial.print("[");
    Serial.print(getDateTime());
    Serial.print("] ");
    
    for (int i = 0; i < 4; i++) {
      int val = analogRead(sensorPins[i]);
      int threshold = baseline[i] * (100.0 - REDUCTION_PERCENT) / 100.0;
      
      Serial.print("S");
      Serial.print(i+1);
      Serial.print(":");
      Serial.print(val);
      
      if (val > (baseline[i] * 0.9)) {
        Serial.print("◯");
      } else if (val < threshold) {
        Serial.print("↓");
      } else {
        Serial.print("-");
      }
      
      if (i < 3) Serial.print(" ");
    }
    
    Serial.print(" | Count: ");
    Serial.println(detectionCount);
    
    static bool alreadyPublished = false;
    
    if (anyDetection && !alreadyPublished) {
      publishData();
      alreadyPublished = true;
    }
    
    if (!anyDetection) {
      alreadyPublished = false;
    }
    lastDisplay = millis();
  }
  
  if (Serial.available()) {
    char c = Serial.read();
    if (c == 'c') {
      calibrateSensors();
    }
  }
}
