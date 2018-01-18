add_library(Net SHARED Net.cpp)
target_compile_definitions(Net PRIVATE MNIST_DATA_LOCATION="${MNIST_DATA_DIR}")

