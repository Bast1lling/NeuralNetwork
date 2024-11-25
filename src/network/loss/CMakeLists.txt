# Declare the math library
add_library(loss
        loss.cpp
        l1.cpp
        l2.cpp
        cross_entropy_loss.cpp
)

# Include directories for this module
target_include_directories(loss PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(loss PRIVATE -Wall -Wextra -Wpedantic -Werror -fsanitize=address)
target_link_options(loss PRIVATE -fsanitize=address)
target_link_libraries(loss PRIVATE math)