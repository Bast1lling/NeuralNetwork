# adopted from https://cliutils.gitlab.io/modern-cmake/chapters/basics/structure.html
cmake_minimum_required(VERSION 3.13)
project(neural-network
        VERSION 0.1.0
        DESCRIPTION "A custom MLP"
        LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(src)
add_subdirectory(test)
