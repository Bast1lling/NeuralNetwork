# Declare the math library
add_library(math vector.cpp matrix.cpp)

# Include directories for this module
target_include_directories(math PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(math PRIVATE -Wall -Wextra -Wpedantic -Werror -fsanitize=address)
target_link_options(math PRIVATE -fsanitize=address)