#pragma once
/**
 * @file hand_tracker.hpp
 * @brief 21-point hand landmark tracker using ESP-DL inference.
 */

#include <array>
#include <cstdint>
#include <memory>

// ESP-DL forward declarations
#include "dl_model_base.hpp"

namespace hand {

/// Number of hand landmarks (wrist + 4 per finger).
constexpr int kNumLandmarks = 21;

/// Model input spatial size (grayscale).
constexpr int kInputSize = 96;

/// A single (x, y) landmark in normalised [0, 1] coordinates.
struct Point2f {
    float x = 0.0f;
    float y = 0.0f;
};

/// Full set of 21 landmarks plus a confidence / validity flag.
struct HandResult {
    bool valid = false;
    std::array<Point2f, kNumLandmarks> landmarks{};
};

/**
 * @class HandTracker
 * @brief Loads the quantised ESP-DL model and runs landmark inference on a
 *        96×96 grayscale image.
 */
class HandTracker {
public:
    HandTracker();
    ~HandTracker();

    // Non-copyable
    HandTracker(const HandTracker&) = delete;
    HandTracker& operator=(const HandTracker&) = delete;

    /**
     * @brief Initialise the model from an embedded ESPDL blob.
     * @param model_data Pointer to the embedded .espdl data.
     * @param model_size Size of the embedded data in bytes.
     * @return true on success.
     */
    bool init(const uint8_t* model_data, size_t model_size);

    /**
     * @brief Run inference on a 96×96 grayscale image.
     * @param gray_96x96 Pointer to row-major uint8 pixel data (96 × 96 bytes).
     * @return HandResult with normalised landmark positions.
     */
    HandResult run(const uint8_t* gray_96x96);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace hand
