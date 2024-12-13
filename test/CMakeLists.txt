add_executable(test main.cpp)

target_link_libraries(test PRIVATE layer math network optimizer loss)

target_compile_options(test PRIVATE -Wall -Wextra -Wpedantic -Werror -fsanitize=address)
target_link_options(test PRIVATE -fsanitize=address)