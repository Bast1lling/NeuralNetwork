add_library(network network.cpp)

target_include_directories(network PUBLIC ../../include)

target_compile_options(network PRIVATE -Wall -Wextra -Wpedantic -Werror -fsanitize=address)
target_link_options(network PRIVATE -fsanitize=address)
