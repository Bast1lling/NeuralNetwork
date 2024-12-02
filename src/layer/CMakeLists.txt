add_library(layer sigmoid.cpp
        layer.cpp
        relu.cpp
        affine.cpp)

target_include_directories(layer PUBLIC ../../include)
target_link_libraries(layer PRIVATE math)

target_compile_options(layer PRIVATE -Wall -Wextra -Wpedantic -Werror -fsanitize=address)
target_link_options(layer PRIVATE -fsanitize=address)