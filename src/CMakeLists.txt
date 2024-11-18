cmake_minimum_required(VERSION 3.13)
project(neural-network)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# find_package(GTest CONFIG REQUIRED)
add_subdirectory(math)
add_subdirectory(network)
add_executable(main main.cpp)
# target_link_libraries(tests GTest::gtest_main)
target_compile_options(main PRIVATE -Wall -Wextra -Wpedantic -Werror -fsanitize=address)
target_link_options(main PRIVATE -fsanitize=address)
target_link_libraries(main PRIVATE network)