add_executable(test main.cpp
        ../include/nn/optimizer/sgd.hpp)

target_link_libraries(test PRIVATE layer loss math network optimizer)

target_compile_options(test PRIVATE -Wall -Wextra -Wpedantic -Werror -fsanitize=address)
target_link_options(test PRIVATE -fsanitize=address)