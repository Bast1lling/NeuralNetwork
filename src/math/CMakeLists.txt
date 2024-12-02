add_library(math vector.cpp matrix.cpp)

target_include_directories(math PUBLIC ../../include)

target_compile_options(math PRIVATE -Wall -Wextra -Wpedantic -Werror -fsanitize=address)
target_link_options(math PRIVATE -fsanitize=address)