add_library(loss
        loss.cpp
        l1.cpp
        l2.cpp
        cross_entropy_loss.cpp
)

target_include_directories(loss PUBLIC ../../include)
target_link_libraries(loss PRIVATE math)

target_compile_options(loss PRIVATE -Wall -Wextra -Wpedantic -Werror -fsanitize=address)
target_link_options(loss PRIVATE -fsanitize=address)