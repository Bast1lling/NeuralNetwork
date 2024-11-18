# Declare the math library
add_library(layer sigmoid.cpp
        layer.cpp)

# Include directories for this module
target_include_directories(layer PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(layer PRIVATE -Wall -Wextra -Wpedantic -Werror -fsanitize=address)
target_link_options(layer PRIVATE -fsanitize=address)
target_link_libraries(layer PUBLIC math)