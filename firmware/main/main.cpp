/**
 * @file main.cpp
 * @brief ESP32-S3 hand landmark tracker – application entry point.
 *
 * Captures frames from the camera, runs ROI detection and landmark
 * inference, then outputs landmark coordinates over UART.
 */

#include <cstdio>
#include <cstring>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "esp_camera.h"

#include "hand_tracker.hpp"

static const char* TAG = "main";

// ---------------------------------------------------------------------------
// Embedded model (linked via CMakeLists.txt EMBED_FILES)
// ---------------------------------------------------------------------------
extern const uint8_t model_start[] asm("_binary_hand_landmarks_espdl_start");
extern const uint8_t model_end[]   asm("_binary_hand_landmarks_espdl_end");

// ---------------------------------------------------------------------------
// Camera pin configuration (ESP-S3-EYE / FREENOVE)
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
// Simple skin-colour ROI (grayscale threshold fallback)
// ---------------------------------------------------------------------------
static bool find_hand_roi(
    const uint8_t* gray, int width, int height,
    int& roi_x, int& roi_y, int& roi_size
) {
    // Coarse threshold-based detection on grayscale
    int min_x = width, min_y = height, max_x = 0, max_y = 0;
    int count = 0;

    for (int y = 0; y < height; y += 2) {
        for (int x = 0; x < width; x += 2) {
            uint8_t px = gray[y * width + x];
            // Simple brightness threshold (tune for your setup)
            if (px > 60 && px < 220) {
                if (x < min_x) min_x = x;
                if (x > max_x) max_x = x;
                if (y < min_y) min_y = y;
                if (y > max_y) max_y = y;
                count++;
            }
        }
    }

    if (count < 100) return false;

    int w = max_x - min_x;
    int h = max_y - min_y;
    roi_size = (w > h) ? w : h;
    roi_x = min_x;
    roi_y = min_y;
    return roi_size > 20;
}

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
// Main task
// ---------------------------------------------------------------------------
static void inference_task(void* /*arg*/) {
    // Initialise camera
    esp_err_t err = esp_camera_init(&camera_config);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Camera init failed: 0x%x", err);
        vTaskDelete(nullptr);
        return;
    }
    ESP_LOGI(TAG, "Camera initialised");

    // Load hand landmark model
    hand::HandTracker tracker;
    size_t model_size = model_end - model_start;
    if (!tracker.init(model_start, model_size)) {
        ESP_LOGE(TAG, "Model init failed");
        vTaskDelete(nullptr);
        return;
    }
    ESP_LOGI(TAG, "Hand tracker ready (model %u bytes)", (unsigned)model_size);

    // Inference loop
    uint32_t frame_count = 0;
    while (true) {
        camera_fb_t* fb = esp_camera_fb_get();
        if (!fb) {
            ESP_LOGW(TAG, "Frame capture failed");
            vTaskDelay(pdMS_TO_TICKS(50));
            continue;
        }

        // The camera is configured for 96×96 grayscale, which matches
        // the model input directly — no resize needed.
        if (fb->width == hand::kInputSize && fb->height == hand::kInputSize) {
            hand::HandResult result = tracker.run(fb->buf);
            if (result.valid) {
                print_landmarks(result);
            }
        } else {
            ESP_LOGW(TAG, "Unexpected frame size: %dx%d", fb->width, fb->height);
        }

        esp_camera_fb_return(fb);
        frame_count++;

        if (frame_count % 100 == 0) {
            ESP_LOGI(TAG, "Frames processed: %lu, free heap: %lu",
                     (unsigned long)frame_count,
                     (unsigned long)esp_get_free_heap_size());
        }

        // Yield to other tasks
        vTaskDelay(pdMS_TO_TICKS(10));
    }
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------
extern "C" void app_main(void) {
    ESP_LOGI(TAG, "=== ESP32-S3 Hand Landmark Tracker ===");
    ESP_LOGI(TAG, "Free heap: %lu bytes", (unsigned long)esp_get_free_heap_size());

    // Run inference on core 1, 8 KB stack
    xTaskCreatePinnedToCore(
        inference_task,
        "inference",
        8192,
        nullptr,
        5,
        nullptr,
        1  // core 1
    );
}
