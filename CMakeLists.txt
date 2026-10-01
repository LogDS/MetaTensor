cmake_minimum_required(VERSION 4.0)
# 1. Forza i percorsi di ricerca di CMake per CUDA prima di trovare i pacchetti

set(CMAKE_CUDA_COMPILER "/usr/local/cuda-13.2/bin/nvcc")
set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-13.2")
set(CUDAToolkit_ROOT "/usr/local/cuda-13.2")
# Indica a nvcc quale compilatore C++ usare dietro le quinte
string(APPEND CMAKE_CUDA_FLAGS " -ccbin /usr/bin/g++-12")

# Configura i compilatori per il resto del progetto C++
set(CMAKE_C_COMPILER "/usr/bin/gcc-12")
set(CMAKE_CXX_COMPILER "/usr/bin/g++-12")

# 2. Forza le variabili d'ambiente direttamente da CMake
set(ENV{CUDA_HOME} "/usr/local/cuda-12.6")
set(ENV{PATH} "/usr/local/cuda-12.6/bin:$ENV{PATH}")
set(ENV{LD_LIBRARY_PATH} "/usr/local/cuda-12.6/lib64:$ENV{LD_LIBRARY_PATH}")

set(Torch_DIR "/home/gyankos/Scaricati/libtorch-shared-with-deps-2.14.0+cu132/libtorch/share/cmake/Torch")
project(MetaTensor LANGUAGES CXX CUDA)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
find_package(Torch REQUIRED)

if (CMAKE_VERSION VERSION_LESS 3.18)
    set(DEV_MODULE Development)
else()
    set(DEV_MODULE Development.Module)
endif()
find_package(Python 3.13 COMPONENTS Interpreter ${DEV_MODULE} REQUIRED)

set(CMAKE_CXX_STANDARD 26)
include_directories(include)
include_directories(submodules/catch2/src)
include_directories(submodules/indicators/include)
add_subdirectory(submodules/catch2)
add_subdirectory(submodules/nanobind)

add_executable(tensor_rank
        examples/tensor_rank.cpp)
target_link_libraries(tensor_rank "${TORCH_LIBRARIES}")
set_property(TARGET tensor_rank PROPERTY CXX_STANDARD 26)

add_executable(some_example tests/simple_tests.cpp)
target_link_libraries(some_example Catch2WithMain "${TORCH_LIBRARIES}")

add_executable(memory_transfer examples/memory_transfer.cpp)
target_link_libraries(memory_transfer "${TORCH_LIBRARIES}")

add_executable(gradient_descent examples/gradient_descent.cpp)
target_link_libraries(gradient_descent "${TORCH_LIBRARIES}")

nanobind_add_module(metatensor_core NOMINSIZE src/cpp/logds/metatensor/bindings.cpp)
target_link_libraries(metatensor_core PRIVATE ${TORCH_LIBRARIES})
install(TARGETS metatensor_core LIBRARY DESTINATION metatensor_py)