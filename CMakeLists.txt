cmake_minimum_required(VERSION 3.26) # scikit-build-core richiede almeno 3.15+, 3.26 garantisce la stabilità per nanobind e CUDA 13

# 1. Uniforma i percorsi di ricerca di CMake per CUDA 13.2 prima di trovare i pacchetti
set(CMAKE_CUDA_COMPILER "/usr/local/cuda-13.2/bin/nvcc")
set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-13.2")
set(CUDAToolkit_ROOT "/usr/local/cuda-13.2")
string(APPEND CMAKE_CUDA_FLAGS " -ccbin /usr/bin/g++-12")

# Configura i compilatori per il resto del progetto C++
set(CMAKE_C_COMPILER "/usr/bin/gcc-12")
set(CMAKE_CXX_COMPILER "/usr/bin/g++-12")

# Corrette le variabili d'ambiente per puntare coerentemente a CUDA 13.2
set(ENV{CUDA_HOME} "/usr/local/cuda-13.2")
set(ENV{PATH} "/usr/local/cuda-13.2/bin:$ENV{PATH}")
set(ENV{LD_LIBRARY_PATH} "/usr/local/cuda-13.2/lib64:$ENV{LD_LIBRARY_PATH}")

set(Torch_DIR "/home/gyankos/Scaricati/libtorch-shared-with-deps-2.14.0+cu132/libtorch/share/cmake/Torch")

# Rimosso CUDA dai linguaggi globali del progetto per evitare la propagazione automatica dello standard 26
project(MetaTensor LANGUAGES CXX)

# Abilita CUDA in modo isolato per i target che lo richiedono senza sporcare il dialetto CXX globale
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    # Blocca esplicitamente lo standard CUDA al massimo supportato da NVCC 13.2 (C++20)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
find_package(Torch REQUIRED)

if (CMAKE_VERSION VERSION_LESS 3.18)
    set(DEV_MODULE Development)
else()
    set(DEV_MODULE Development.Module)
endif()
find_package(Python 3.13 COMPONENTS Interpreter ${DEV_MODULE} REQUIRED)

# RIMOSSO: set(CMAKE_CXX_STANDARD 26) <- Questa riga rompeva NVCC

include_directories(include)
include_directories(submodules/catch2/src)
include_directories(submodules/indicators/include)
add_subdirectory(submodules/catch2)
add_subdirectory(submodules/nanobind)

# --- Configurazione dei singoli Target con isolamento C++26 (o C++2c per GCC 12) ---

# Helper macro per forzare il massimo standard disponibile su GCC 12 senza rompere CMake
macro(apply_maximum_cxx_standard TARGET_NAME)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS "14.0")
        # Su GCC 12, C++26 non è nativo in CMake: usiamo il flag sperimentale diretto del compilatore
        target_compile_options(${TARGET_NAME} PRIVATE "-std=gnu++20")
    else()
        # Se in futuro aggiornerai a GCC 14+, CMake userà nativamente le astrazioni moderne
        target_compile_features(${TARGET_NAME} PRIVATE cxx_std_26)
    endif()
endmacro()

add_executable(tensor_rank examples/tensor_rank.cpp)
target_link_libraries(tensor_rank "${TORCH_LIBRARIES}")
apply_maximum_cxx_standard(tensor_rank)

add_executable(some_example tests/simple_tests.cpp)
target_link_libraries(some_example Catch2WithMain "${TORCH_LIBRARIES}")
apply_maximum_cxx_standard(some_example)

add_executable(memory_transfer examples/memory_transfer.cpp)
target_link_libraries(memory_transfer "${TORCH_LIBRARIES}")
apply_maximum_cxx_standard(memory_transfer)

add_executable(gradient_descent examples/gradient_descent.cpp)
target_link_libraries(gradient_descent "${TORCH_LIBRARIES}")
apply_maximum_cxx_standard(gradient_descent)

# --- Modulo Nanobind ---
nanobind_add_module(metatensor_core NOMINSIZE src/cpp/logds/metatensor/bindings.cpp)
target_link_libraries(metatensor_core PRIVATE ${TORCH_LIBRARIES})
apply_maximum_cxx_standard(metatensor_core) # Forzerà la modalità C++2c/26 abilitando i fold expression

install(TARGETS metatensor_core LIBRARY DESTINATION .)
