/**
 * @file main.cpp
 * @brief ESP32-S3 hand landmark tracker – application entry point.
 *
 * Captures frames from the camera, runs landmark inference,
 * and outputs landmark coordinates as JSON over UART.
 */

#include <cstdio>
#include <cstring>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "esp_system.h"
#include "esp_camera.h"

#include "hand_tracker.hpp"

static const char* TAG = "main";

// ---------------------------------------------------------------------------
// Embedded model symbols (linked via CMakeLists.txt EMBED_FILES)
// ---------------------------------------------------------------------------
// Some ESP-IDF versions strip path prefix while others retain "models_"
extern const uint8_t model_start[] asm("_binary_hand_landmarks_espdl_start") __attribute__((weak));
extern const uint8_t model_end[]   asm("_binary_hand_landmarks_espdl_end") __attribute__((weak));

extern const uint8_t model_rel_start[] asm("_binary_models_hand_landmarks_espdl_start") __attribute__((weak));
extern const uint8_t model_rel_end[]   asm("_binary_models_hand_landmarks_espdl_end") __attribute__((weak));

static inline void get_embedded_model(const uint8_t*& start, size_t& size) {
    if (model_rel_start != nullptr && model_rel_end != nullptr && model_rel_end > model_rel_start) {
        start = model_rel_start;
        size = model_rel_end - model_rel_start;
    } else if (model_start != nullptr && model_end != nullptr && model_end > model_start) {
        start = model_start;
        size = model_end - model_start;
    } else {
        start = nullptr;
        size = 0;
    }
}

// ---------------------------------------------------------------------------
// Camera pin configuration (ESP-S3-EYE / FREENOVE default pins)
// ---------------------------------------------------------------------------
static const camera_config_t camera_config = {
    .pin_pwdn     = -1,
    .pin_reset    = -1,
    .pin_xclk     = 15,
    .pin_sccb_sda = 4,
    .pin_sccb_scl = 5,

    .pin_d7 = 16,
    .pin_d6 = 17,
    .pin_d5 = 18,
    .pin_d4 = 12,
    .pin_d3 = 10,
    .pin_d2 = 8,
    .pin_d1 = 9,
    .pin_d0 = 11,

    .pin_vsync = 6,
    .pin_href  = 7,
    .pin_pclk  = 13,

    .xclk_freq_hz = 20000000,
    .ledc_timer   = LEDC_TIMER_0,
    .ledc_channel = LEDC_CHANNEL_0,

    .pixel_format = PIXFORMAT_GRAYSCALE,
    .frame_size   = FRAMESIZE_96X96,
    .jpeg_quality = 12,
    .fb_count     = 2,
    .fb_location  = CAMERA_FB_IN_PSRAM,
    .grab_mode    = CAMERA_GRAB_LATEST,
};

// ---------------------------------------------------------------------------
// Print landmarks as JSON over UART
// ---------------------------------------------------------------------------
static void print_landmarks(const hand::HandResult& result) {
    printf("{\"landmarks\":[");
    for (int i = 0; i < hand::kNumLandmarks; ++i) {
        if (i > 0) printf(",");
        printf("[%.4f,%.4f]", result.landmarks[i].x, result.landmarks[i].y);
    }
    printf("]}\n");
}

// ---------------------------------------------------------------------------
// Main inference task
// ---------------------------------------------------------------------------
static void inference_task(void* /*arg*/) {
    // Initialise camera
    esp_err_t err = esp_camera_init(&camera_config);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Camera init failed: 0x%x. Verify pinout and sensor connection.", err);
        vTaskDelete(nullptr);
        return;
    }
    ESP_LOGI(TAG, "Camera initialised (96x96 grayscale)");

    // Retrieve embedded model
    const uint8_t* model_data = nullptr;
    size_t model_size = 0;
    get_embedded_model(model_data, model_size);

    hand::HandTracker tracker;
    if (!model_data || model_size == 0 || !tracker.init(model_data, model_size)) {
        ESP_LOGW(TAG, "Model not loaded (size=%u). Running camera stream only.", (unsigned)model_size);
        ESP_LOGW(TAG, "Train and quantize model using python/ scripts, then re-flash.");
    } else {
        ESP_LOGI(TAG, "Hand tracker ready (%u bytes model)", (unsigned)model_size);
    }

    // Inference loop
    uint32_t frame_count = 0;
    while (true) {
        camera_fb_t* fb = esp_camera_fb_get();
        if (!fb) {
            ESP_LOGW(TAG, "Frame capture failed");
            vTaskDelay(pdMS_TO_TICKS(50));
            continue;
        }

        if (fb->width == hand::kInputSize && fb->height == hand::kInputSize) {
            hand::HandResult result = tracker.run(fb->buf);
            if (result.valid) {
                print_landmarks(result);
            }
        } else {
            ESP_LOGW(TAG, "Unexpected frame size: %dx%d (expected %dx%d)",
                     fb->width, fb->height, hand::kInputSize, hand::kInputSize);
        }

        esp_camera_fb_return(fb);
        frame_count++;

        if (frame_count % 100 == 0) {
            ESP_LOGI(TAG, "Frames processed: %lu, free heap: %lu bytes",
                     (unsigned long)frame_count,
                     (unsigned long)esp_get_free_heap_size());
        }

        vTaskDelay(pdMS_TO_TICKS(10));
    }
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------
extern "C" void app_main(void) {
    ESP_LOGI(TAG, "=== ESP32-S3 Hand Landmark Tracker ===");
    ESP_LOGI(TAG, "Free heap: %lu bytes", (unsigned long)esp_get_free_heap_size());

    // Run inference task on Core 1 with 8 KB stack
    xTaskCreatePinnedToCore(
        inference_task,
        "inference",
        8192,
        nullptr,
        5,
        nullptr,
        1
    );
}
