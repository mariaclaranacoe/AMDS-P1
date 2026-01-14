#include <WiFi.h>
#include <PubSubClient.h>
#include <ArduinoJson.h>
#include <time.h>

// WiFi credentials
const char* ssid = "SBG6700AC-69159";
const char* password = "fb13122db2";

// MQTT Broker settings
const char* mqtt_server = "192.168.0.22";
const int mqtt_port = 1883;
const char* mqtt_user = "teasis";
const char* mqtt_password = "teasis";

// MQTT Topics - SINGLE TOPIC NOW
const char* topic_data = "/mosquito/data";  // Single topic for all data

// Sensor pins
const int sensorPins[6] = {34, 35, 32, 33, 25, 26};
bool prevState[6] = {0};
int detectionCount = 0;
bool anyDetection = false;

// NTP Settings
const char* ntpServer = "pool.ntp.org";
const long gmtOffset_sec = 8 * 3600;  // GMT+8 for Philippines
const int daylightOffset_sec = 0;

// Initialize clients
WiFiClient espClient;
PubSubClient client(espClient);

// Variables for timing
unsigned long lastSensorMsg = 0;
const long sensorPublishInterval = 1000;    // Data every 1 second on detection
unsigned long lastRead = 0;
const long readInterval = 50;
bool timeSynced = false;

void setup_wifi() {
  delay(10);
  Serial.println();
  Serial.print("Connecting to ");
  Serial.println(ssid);

  WiFi.begin(ssid, password);

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  Serial.println("");
  Serial.println("WiFi connected");
  Serial.print("IP address: ");
  Serial.println(WiFi.localIP());
}

void reconnect() {
  while (!client.connected()) {
    Serial.print("Attempting MQTT connection...");
    
    if (client.connect("ESP32_IR_Sensor", mqtt_user, mqtt_password)) {
      Serial.println("connected");
    } else {
      Serial.print("failed, rc=");
      Serial.print(client.state());
      Serial.println(" try again in 5 seconds");
      delay(5000);
    }
  }
}

// Get current date and time as string
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

void setup() {
  Serial.begin(115200);
  delay(1000);
  
  Serial.println("\n=== IR BREAK-BEAM SENSOR ===");
  Serial.println("Circuit: Photodiode cathode->3.3V, anode->pin->10k->GND");
  Serial.println("Logic: HIGH=Light detected, LOW=No light");
  Serial.println("MQTT Topic:");
  Serial.println("  /mosquito/data - All data in one message");
  
  // Initialize all pins as digital inputs WITHOUT PULLUP
  for (int i = 0; i < 6; i++) {
    pinMode(sensorPins[i], INPUT);  // NO PULLUP - you have external resistor!
    
    // Read initial state
    bool rawReading = digitalRead(sensorPins[i]);
    prevState[i] = rawReading;
    
    Serial.print("Sensor ");
    Serial.print(i+1);
    Serial.print(" (Pin ");
    Serial.print(sensorPins[i]);
    Serial.print("): ");
    Serial.print(rawReading ? "HIGH" : "LOW");
    Serial.print(" = ");
    Serial.println(rawReading ? "LIGHT DETECTED" : "NO LIGHT");
    
    delay(100);
  }
  
  Serial.println("=== SETUP COMPLETE ===");
  
  setup_wifi();
  
  // Initialize and sync time via NTP
  Serial.println("Syncing time via NTP...");
  configTime(gmtOffset_sec, daylightOffset_sec, ntpServer);
  
  // Wait for time to be synced
  struct tm timeinfo;
  for (int i = 0; i < 20; i++) {  // Try for 10 seconds
    if (getLocalTime(&timeinfo)) {
      timeSynced = true;
      char datetime[20];
      strftime(datetime, sizeof(datetime), "%Y-%m-%d %H:%M:%S", &timeinfo);
      Serial.print("Time synced: ");
      Serial.println(datetime);
      break;
    }
    Serial.print(".");
    delay(500);
  }
  
  if (!timeSynced) {
    Serial.println("Failed to sync time");
  }
  
  client.setServer(mqtt_server, mqtt_port);
  client.setBufferSize(256);
}

void readSensors() {
  anyDetection = false;
  
  for (int i = 0; i < 6; i++) {
    // Read digital value
    bool currentReading = digitalRead(sensorPins[i]);
    
    // Check for state change from HIGH to LOW (beam broken)
    if (currentReading == 0 && prevState[i] == 1) {
      detectionCount++;
      anyDetection = true;
      
      // Get current time for the detection
      String detectionTime = getDateTime();
      Serial.print("\nDETECTION! Sensor ");
      Serial.print(i+1);
      Serial.print(" (beam broken) at ");
      Serial.println(detectionTime);
      Serial.print("Total mosquito count: ");
      Serial.println(detectionCount);
    }
    
    // Update previous state
    prevState[i] = currentReading;
  }
}

void displayStatus() {
  static unsigned long lastDisplay = 0;
  
  if (anyDetection || (millis() - lastDisplay > 2000)) {
    Serial.print("[");
    Serial.print(getDateTime());
    Serial.print("] ");
    
    for (int i = 0; i < 6; i++) {
      bool rawReading = digitalRead(sensorPins[i]);
      
      Serial.print("s");
      Serial.print(i+1);
      Serial.print(":");
      Serial.print(rawReading ? "HIGH" : "LOW");
      Serial.print("(");
      Serial.print(rawReading ? "INTACT" : "BROKEN");
      Serial.print(")");
      
      if (i < 5) Serial.print(" | ");
    }
    
    Serial.print(" | Total: ");
    Serial.println(detectionCount);
    lastDisplay = millis();
  }
}

void publishAllData() {
  StaticJsonDocument<256> jsonDoc;
  
  // Sensor states (1=beam intact, 0=beam broken)
  jsonDoc["s1"] = digitalRead(sensorPins[0]);
  jsonDoc["s2"] = digitalRead(sensorPins[1]);
  jsonDoc["s3"] = digitalRead(sensorPins[2]);
  jsonDoc["s4"] = digitalRead(sensorPins[3]);
  jsonDoc["s5"] = digitalRead(sensorPins[4]);
  jsonDoc["s6"] = digitalRead(sensorPins[5]);
  jsonDoc["detection_count"] = detectionCount;
  jsonDoc["total_count"] = detectionCount;  // Same value, for compatibility
  jsonDoc["datetime"] = getDateTime();
  
  char jsonBuffer[256];
  serializeJson(jsonDoc, jsonBuffer);
  
  // Publish to SINGLE topic only
  if (client.publish(topic_data, jsonBuffer)) {
    Serial.print("MQTT Published to /mosquito/data: ");
    Serial.println(jsonBuffer);
  }
}

void loop() {
  if (!client.connected()) {
    reconnect();
  }
  client.loop();
  
  unsigned long now = millis();
  
  // Read sensors every 50ms
  if (now - lastRead >= readInterval) {
    readSensors();
    lastRead = now;
  }
  
  // Publish ALL data when detection occurs
  if (anyDetection) {
    displayStatus();
    
    if (now - lastSensorMsg >= sensorPublishInterval) {
      publishAllData();    // Send everything in one message
      lastSensorMsg = now;
    }
  }
  
  // Show connection status every 30 seconds
  static unsigned long lastStatus = 0;
  if (now - lastStatus > 30000) {
    Serial.print("[");
    Serial.print(getDateTime());
    Serial.print("] Status: WiFi=");
    Serial.print(WiFi.status() == WL_CONNECTED ? "OK" : "NO");
    Serial.print(" | MQTT=");
    Serial.println(client.connected() ? "OK" : "NO");
    lastStatus = now;
  }
}
