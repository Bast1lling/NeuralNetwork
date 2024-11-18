# Declare the math library
add_library(network network.cpp)

add_subdirectory(layer)
# Include directories for this module
target_include_directories(network PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(network PRIVATE -Wall -Wextra -Wpedantic -Werror -fsanitize=address)
target_link_options(network PRIVATE -fsanitize=address)
target_link_libraries(network
        PUBLIC layer
)
