set(embed_files "")
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/models/hand_landmarks.espdl")
    list(APPEND embed_files "models/hand_landmarks.espdl")
endif()

idf_component_register(
    SRCS
        "main.cpp"
        "hand_tracker.cpp"
    INCLUDE_DIRS
        "."
    EMBED_FILES
        ${embed_files}
)
