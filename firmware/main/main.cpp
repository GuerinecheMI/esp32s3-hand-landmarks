/**
 * @file main.cpp
 * @brief ESP32-S3 dynamic hand landmark tracker with adaptive ROI scanning.
 *
 * Captures 160x120 QQVGA frames, dynamically tracks and centers the hand
 * anywhere in the field of view, crops/scales to 96x96, and runs ESP-DL inference.
 */

#include <cstdio>
#include <cstring>
#include <cmath>
#include <algorithm>

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
extern const uint8_t model_start[]     asm("_binary_hand_landmarks_espdl_start") __attribute__((weak));
extern const uint8_t model_end[]       asm("_binary_hand_landmarks_espdl_end") __attribute__((weak));

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
// Camera configuration: 160x120 QQVGA for wide field-of-view scanning
// ---------------------------------------------------------------------------
static constexpr int CAM_W = 160;
static constexpr int CAM_H = 120;

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
    .frame_size   = FRAMESIZE_QQVGA,  // 160 x 120
    .jpeg_quality = 12,
    .fb_count     = 2,
    .fb_location  = CAMERA_FB_IN_PSRAM,
    .grab_mode    = CAMERA_GRAB_LATEST,
};

// ---------------------------------------------------------------------------
// Adaptive ROI tracker state
// ---------------------------------------------------------------------------
struct RoiTracker {
    float x = 32.0f;     // Top-left x
    float y = 12.0f;     // Top-left y
    float size = 96.0f;  // Square ROI dimension
    bool tracked = false;
    int lost_count = 0;
};

// Fast integer crop and scale from ROI -> 96x96
static void crop_and_rescale(
    const uint8_t* src, int src_w, int src_h,
    int rx, int ry, int rsize,
    uint8_t* dst, int dst_size
) {
    // Fixed-point scaling ratio (16-bit shift)
    int step = (rsize << 16) / dst_size;

    for (int dy = 0; dy < dst_size; ++dy) {
        int sy = ry + ((dy * step) >> 16);
        if (sy < 0) sy = 0;
        if (sy >= src_h) sy = src_h - 1;

        const uint8_t* src_row = src + sy * src_w;
        uint8_t* dst_row = dst + dy * dst_size;

        for (int dx = 0; dx < dst_size; ++dx) {
            int sx = rx + ((dx * step) >> 16);
            if (sx < 0) sx = 0;
            if (sx >= src_w) sx = src_w - 1;
            dst_row[dx] = src_row[sx];
        }
    }
}

// Full frame scanner when tracking is lost: finds hand cluster in 160x120
static bool scan_for_hand(const uint8_t* frame, int w, int h, RoiTracker& roi) {
    int sum_x = 0;
    int sum_y = 0;
    int count = 0;

    // Subsampled scan (step of 3)
    for (int y = 8; y < h - 8; y += 3) {
        for (int x = 8; x < w - 8; x += 3) {
            uint8_t px = frame[y * w + x];
            // Typical indoor skin/hand intensity window
            if (px > 65 && px < 215) {
                sum_x += x;
                sum_y += y;
                count++;
            }
        }
    }

    if (count > 60) {
        float cx = static_cast<float>(sum_x) / count;
        float cy = static_cast<float>(sum_y) / count;
        float new_size = 88.0f;

        roi.x = std::clamp(cx - new_size / 2.0f, 0.0f, static_cast<float>(w - new_size));
        roi.y = std::clamp(cy - new_size / 2.0f, 0.0f, static_cast<float>(h - new_size));
        roi.size = new_size;
        roi.tracked = true;
        roi.lost_count = 0;
        return true;
    }

    return false;
}

// ---------------------------------------------------------------------------
// Print full results as compact JSON over UART
// ---------------------------------------------------------------------------
static void print_landmarks(
    const hand::HandResult& result,
    const RoiTracker& roi,
    int cam_w, int cam_h
) {
    // Output full camera normalized coords + local crop coords + tracking box
    printf("{\"roi\":[%.1f,%.1f,%.1f],\"landmarks\":[",
           roi.x, roi.y, roi.size);

    for (int i = 0; i < hand::kNumLandmarks; ++i) {
        if (i > 0) printf(",");
        // Full frame mapped coordinates [0, 1]
        float fx = (roi.x + result.landmarks[i].x * roi.size) / cam_w;
        float fy = (roi.y + result.landmarks[i].y * roi.size) / cam_h;
        printf("[%.4f,%.4f]", fx, fy);
    }
    printf("]}\n");
}

// ---------------------------------------------------------------------------
// Inference Task
// ---------------------------------------------------------------------------
static void inference_task(void* /*arg*/) {
    // Initialise camera in 160x120 QQVGA mode
    esp_err_t err = esp_camera_init(&camera_config);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Camera init failed: 0x%x. Verify pinout and sensor.", err);
        vTaskDelete(nullptr);
        return;
    }
    ESP_LOGI(TAG, "Camera active (160x120 QQVGA wide-view mode)");

    // Retrieve embedded model
    const uint8_t* model_data = nullptr;
    size_t model_size = 0;
    get_embedded_model(model_data, model_size);

    hand::HandTracker tracker;
    if (!model_data || model_size == 0 || !tracker.init(model_data, model_size)) {
        ESP_LOGW(TAG, "Model not loaded (size=%u). Check embedded .espdl file.", (unsigned)model_size);
    } else {
        ESP_LOGI(TAG, "Hand tracker ready (%u bytes model)", (unsigned)model_size);
    }

    // Allocate 96x96 crop buffer in internal RAM
    uint8_t* crop_buf = static_cast<uint8_t*>(
        heap_caps_malloc(hand::kInputSize * hand::kInputSize, MALLOC_CAP_8BIT | MALLOC_CAP_INTERNAL)
    );
    if (!crop_buf) {
        crop_buf = static_cast<uint8_t*>(malloc(hand::kInputSize * hand::kInputSize));
    }

    RoiTracker roi;
    uint32_t frame_count = 0;

    while (true) {
        camera_fb_t* fb = esp_camera_fb_get();
        if (!fb) {
            ESP_LOGW(TAG, "Frame grab failed");
            vTaskDelay(pdMS_TO_TICKS(40));
            continue;
        }

        // If tracking is lost, scan full 160x120 scene for hand position
        if (!roi.tracked) {
            scan_for_hand(fb->buf, fb->width, fb->height, roi);
        }

        // Clamp ROI within camera frame bounds
        int rx = std::clamp(static_cast<int>(roi.x), 0, fb->width - hand::kInputSize);
        int ry = std::clamp(static_cast<int>(roi.y), 0, fb->height - hand::kInputSize);
        int rsize = std::clamp(static_cast<int>(roi.size), 48, std::min(fb->width, fb->height));
        if (rx + rsize > fb->width)  rx = fb->width - rsize;
        if (ry + rsize > fb->height) ry = fb->height - rsize;

        // Crop & downscale to 96x96 for model
        crop_and_rescale(fb->buf, fb->width, fb->height, rx, ry, rsize, crop_buf, hand::kInputSize);

        // Run ESP-DL landmark inference
        hand::HandResult result = tracker.run(crop_buf);

        if (result.valid) {
            print_landmarks(result, roi, fb->width, fb->height);

            // Compute landmark bounding box extent inside the crop [0, 1]
            float min_x = 1.0f, max_x = 0.0f;
            float min_y = 1.0f, max_y = 0.0f;
            for (int i = 0; i < hand::kNumLandmarks; ++i) {
                min_x = std::min(min_x, result.landmarks[i].x);
                max_x = std::max(max_x, result.landmarks[i].x);
                min_y = std::min(min_y, result.landmarks[i].y);
                max_y = std::max(max_y, result.landmarks[i].y);
            }

            float span_w = max_x - min_x;
            float span_h = max_y - min_y;

            // If landmark geometry is consistent, smoothly track the hand's center
            if (span_w > 0.12f && span_h > 0.12f && span_w < 0.95f && span_h < 0.95f) {
                float center_crop_x = (min_x + max_x) * 0.5f;
                float center_crop_y = (min_y + max_y) * 0.5f;

                float new_cam_cx = roi.x + center_crop_x * roi.size;
                float new_cam_cy = roi.y + center_crop_y * roi.size;
                float desired_size = std::max(span_w, span_h) * roi.size * 1.35f;
                desired_size = std::clamp(desired_size, 64.0f, 115.0f);

                // Smooth filtering (EMA) to prevent jitter
                roi.size = 0.70f * roi.size + 0.30f * desired_size;
                float target_x = new_cam_cx - roi.size * 0.5f;
                float target_y = new_cam_cy - roi.size * 0.5f;

                roi.x = 0.65f * roi.x + 0.35f * target_x;
                roi.y = 0.65f * roi.y + 0.35f * target_y;

                roi.x = std::clamp(roi.x, 0.0f, static_cast<float>(fb->width - roi.size));
                roi.y = std::clamp(roi.y, 0.0f, static_cast<float>(fb->height - roi.size));

                roi.tracked = true;
                roi.lost_count = 0;
            } else {
                roi.lost_count++;
                if (roi.lost_count > 6) {
                    roi.tracked = false;  // Fallback to full-frame scanning
                }
            }
        }

        esp_camera_fb_return(fb);
        frame_count++;

        if (frame_count % 150 == 0) {
            ESP_LOGI(TAG, "Frames: %lu | Free Heap: %lu | ROI: [%.0f, %.0f, %.0f]",
                     (unsigned long)frame_count,
                     (unsigned long)esp_get_free_heap_size(),
                     roi.x, roi.y, roi.size);
        }

        vTaskDelay(pdMS_TO_TICKS(10));
    }
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------
extern "C" void app_main(void) {
    ESP_LOGI(TAG, "=== ESP32-S3 Adaptive Hand Tracker ===");
    ESP_LOGI(TAG, "Free heap: %lu bytes", (unsigned long)esp_get_free_heap_size());

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
