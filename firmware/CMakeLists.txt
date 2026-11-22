# ESP-IDF top-level CMakeLists.txt
cmake_minimum_required(VERSION 3.16)

set(EXTRA_COMPONENT_DIRS "main")

include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(hand_landmarks)
