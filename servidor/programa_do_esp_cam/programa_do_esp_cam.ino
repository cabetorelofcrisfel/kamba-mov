#include "esp_camera.h"
#include <WiFi.h>
#include <ESPAsyncWebServer.h>
#include <HTTPClient.h>

// ================= WIFI =================
const char* ssid = "vivo";
const char* password = "12345678";

// ================= PINAGEM AI THINKER =================
#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27
#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22

AsyncWebServer server(80);

bool sendToAPI = false;
String apiURL = "https://kamba-mov.vercel.app/upload";

// ===== PROTÓTIPOS =====
void handleRoot(AsyncWebServerRequest *request);
void handleStreamPage(AsyncWebServerRequest *request);
void handleStream(AsyncWebServerRequest *request);
void handleToggleAPI(AsyncWebServerRequest *request);
void handleSetAPI(AsyncWebServerRequest *request);

void setup() {
  Serial.begin(115200);

  // ================= CAMERA =================
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;

  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;

  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;

  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;

  config.xclk_freq_hz = 10000000;
  config.pixel_format = PIXFORMAT_JPEG;

  if(psramFound()){
    config.frame_size = FRAMESIZE_QVGA;
    config.jpeg_quality = 10;
    config.fb_count = 2;
  } else {
    config.frame_size = FRAMESIZE_QQVGA;
    config.jpeg_quality = 12;
    config.fb_count = 1;
  }

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Erro ao iniciar câmera: 0x%x\n", err);
    return;
  }

  // ================= WIFI =================
  WiFi.begin(ssid, password);
  Serial.print("Conectando");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  Serial.println("\nConectado!");
  Serial.print("IP do ESP32-CAM: ");
  Serial.println(WiFi.localIP());

  // ================= ROTAS =================
  server.on("/", HTTP_GET, handleRoot);
  server.on("/streamPage", HTTP_GET, handleStreamPage);
  server.on("/stream", HTTP_GET, handleStream);
  server.on("/toggleAPI", HTTP_POST, handleToggleAPI);
  server.on("/setAPI", HTTP_POST, handleSetAPI);

  server.begin();
}

void loop() {
  delay(1);
}

// ================= PÁGINA PRINCIPAL =================
void handleRoot(AsyncWebServerRequest *request) {
  String html = R"rawliteral(
  <html>
  <head>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    body { background:#111; color:white; text-align:center; font-family:Arial; }
    button { padding:10px; margin:5px; font-size:16px; }
    img { width:95%; max-width:600px; margin-top:20px; }
  </style>
  </head>
  <body>
    <h2>ESP32-CAM Kamba-Mov</h2>

    <a href="/streamPage"><button>Ver Stream</button></a><br><br>

    <form action="/toggleAPI" method="POST">
      <button type="submit">Alternar Envio API</button>
    </form>

    <form action="/setAPI" method="POST">
      <input name="api" placeholder="https://nova-api.com/upload">
      <br><br>
      <button type="submit">Definir Nova API</button>
    </form>

    <p>Status API: )rawliteral";

  html += sendToAPI ? "ATIVADO" : "DESATIVADO";
  html += R"rawliteral(</p>
    <p>API atual:</p>
    <p>)rawliteral";
  html += apiURL;
  html += R"rawliteral(</p>
  </body>
  </html>)rawliteral";

  request->send(200, "text/html", html);
}

// ================= PÁGINA DO STREAM =================
void handleStreamPage(AsyncWebServerRequest *request) {
  String html = R"rawliteral(
  <html>
  <head>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
      body { background:#000; text-align:center; }
      img { width:95%; max-width:600px; margin-top:20px; }
    </style>
    <script>
      function updateImage() {
        var img = document.getElementById("cam");
        img.src = "/stream?time=" + new Date().getTime();
      }
      setInterval(updateImage, 100); // 100 ms = 0,1 s
    </script>
  </head>
  <body>
    <h2 style="color:white;">Stream com Refresh 0,1s</h2>
    <img id="cam" src="/stream">
    <br><br>
    <a href="/"><button>Voltar</button></a>
  </body>
  </html>)rawliteral";

  request->send(200, "text/html", html);
}

// ================= SNAPSHOT =================
void handleStream(AsyncWebServerRequest *request) {
  camera_fb_t * fb = esp_camera_fb_get();
  if (!fb) {
    request->send(500, "text/plain", "Erro ao capturar frame");
    return;
  }

  AsyncWebServerResponse *response = request->beginResponse_P(200, "image/jpeg", fb->buf, fb->len);
  response->addHeader("Content-Disposition", "inline; filename=capture.jpg");
  request->send(response);

  if (sendToAPI) {
    HTTPClient http;
    http.begin(apiURL);
    http.addHeader("Content-Type", "image/jpeg");
    http.POST(fb->buf, fb->len);
    http.end();
  }

  esp_camera_fb_return(fb);
}

// ================= CONTROLES =================
void handleToggleAPI(AsyncWebServerRequest *request) {
  sendToAPI = !sendToAPI;
  request->redirect("/");
}

void handleSetAPI(AsyncWebServerRequest *request) {
  if(request->hasParam("api", true)){
    apiURL = request->getParam("api", true)->value();
  }
  request->redirect("/");
}
