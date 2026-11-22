/**
 * @file hand_tracker.cpp
 * @brief ESP-DL based hand landmark inference implementation.
 */

#include "hand_tracker.hpp"

#include <cstring>
#include <cmath>

#include "esp_log.h"
#include "dl_model_base.hpp"
#include "dl_tool.hpp"

static const char* TAG = "hand_tracker";

namespace hand {

// ---------------------------------------------------------------------------
// Private implementation
// ---------------------------------------------------------------------------
struct HandTracker::Impl {
    dl::Model* model = nullptr;
    // Pre-allocated input buffer (quantised INT8)
    int8_t* input_buf = nullptr;
    float input_scale = 1.0f;
    int input_zero_point = 0;
};

// ---------------------------------------------------------------------------
// Construction / destruction
// ---------------------------------------------------------------------------
HandTracker::HandTracker() : impl_(std::make_unique<Impl>()) {}
HandTracker::~HandTracker() {
    if (impl_ && impl_->model) {
        delete impl_->model;
    }
    if (impl_ && impl_->input_buf) {
        free(impl_->input_buf);
    }
}

// ---------------------------------------------------------------------------
// Initialisation
// ---------------------------------------------------------------------------
bool HandTracker::init(const uint8_t* model_data, size_t model_size) {
    if (!model_data || model_size == 0) {
        ESP_LOGE(TAG, "Invalid model data");
        return false;
    }

    // Load ESP-DL model from memory
    impl_->model = new dl::Model();
    if (!impl_->model->load(model_data, model_size)) {
        ESP_LOGE(TAG, "Failed to load ESP-DL model");
        delete impl_->model;
        impl_->model = nullptr;
        return false;
    }

    // Allocate quantised input buffer
    const size_t input_bytes = kInputSize * kInputSize;
    impl_->input_buf = static_cast<int8_t*>(heap_caps_malloc(input_bytes, MALLOC_CAP_SPIRAM));
    if (!impl_->input_buf) {
        ESP_LOGE(TAG, "Failed to allocate input buffer (%d bytes)", (int)input_bytes);
        return false;
    }

    // Retrieve quantisation parameters from first input tensor
    auto* input_tensor = impl_->model->get_input(0);
    if (input_tensor) {
        impl_->input_scale = input_tensor->get_scale();
        impl_->input_zero_point = input_tensor->get_zero_point();
    }

    ESP_LOGI(TAG, "Model loaded successfully (input scale=%.4f, zp=%d)",
             impl_->input_scale, impl_->input_zero_point);
    return true;
}

// ---------------------------------------------------------------------------
// Inference
// ---------------------------------------------------------------------------
HandResult HandTracker::run(const uint8_t* gray_96x96) {
    HandResult result;

    if (!impl_->model || !impl_->input_buf || !gray_96x96) {
        ESP_LOGW(TAG, "Tracker not initialised or null input");
        return result;
    }

    // Quantise input: int8 = round(float_val / scale) + zero_point
    const float inv_scale = 1.0f / impl_->input_scale;
    const int zp = impl_->input_zero_point;
    const size_t num_pixels = kInputSize * kInputSize;

    for (size_t i = 0; i < num_pixels; ++i) {
        float normalised = gray_96x96[i] / 255.0f;
        int32_t q = static_cast<int32_t>(roundf(normalised * inv_scale)) + zp;
        impl_->input_buf[i] = static_cast<int8_t>(
            q < -128 ? -128 : (q > 127 ? 127 : q));
    }

    // Copy into model input tensor
    auto* input_tensor = impl_->model->get_input(0);
    if (input_tensor) {
        std::memcpy(input_tensor->get_data(), impl_->input_buf, num_pixels);
    }

    // Run forward pass
    impl_->model->run();

    // Extract output: 42 values (21 landmarks × 2 coords)
    auto* output_tensor = impl_->model->get_output(0);
    if (!output_tensor) {
        ESP_LOGW(TAG, "No output tensor");
        return result;
    }

    const float out_scale = output_tensor->get_scale();
    const int out_zp = output_tensor->get_zero_point();
    const int8_t* out_data = static_cast<const int8_t*>(output_tensor->get_data());

    result.valid = true;
    for (int i = 0; i < kNumLandmarks; ++i) {
        // Dequantise: float = (int8 - zero_point) * scale
        float x = (out_data[i * 2 + 0] - out_zp) * out_scale;
        float y = (out_data[i * 2 + 1] - out_zp) * out_scale;
        result.landmarks[i] = {
            .x = fminf(fmaxf(x, 0.0f), 1.0f),
            .y = fminf(fmaxf(y, 0.0f), 1.0f),
        };
    }

    return result;
}

}  // namespace hand
