add_library(optimizer sgd.cpp
        optimizer.cpp)

target_include_directories(optimizer PUBLIC ../../include)
target_link_libraries(optimizer PRIVATE math )

target_compile_options(optimizer PRIVATE -Wall -Wextra -Wpedantic -Werror -fsanitize=address)
target_link_options(optimizer PRIVATE -fsanitize=address)