cmake_minimum_required(VERSION 3.26) # scikit-build-core richiede almeno 3.15+, 3.26 garantisce la stabilità per nanobind e CUDA 13
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
# Opzione per attivare o disattivare il supporto distribuito (Disattivata di default)
option(METATENSOR_USE_MPI "Build MetaTensor with multi-node MPI cluster support" OFF)
add_compile_definitions(AT_PER_OPERATOR_HEADERS)
add_compile_definitions(CAFFE2_USE_GOOGLE_GLOG)
set(CMAKE_CXX_EXTENSIONS ON) # Abilita gnu++ rispetto a c++ puro, risolve i conflitti di typeid.h


if(METATENSOR_USE_MPI)
    # Se attivata dall'utente, iniettiamo la macro di compilazione globale
    add_compile_definitions(METATENSOR_USE_MPI)
    message(STATUS "[MetaTensor] Multi-node Distributed Parallelism ENABLED")
else()
    message(STATUS "[MetaTensor] Running in Single-Machine Local mode (Distributed disabled)")
endif()

# 1. Tentativo di trovare libgccjit usando i percorsi di sistema standard
find_path(LIBGCCJIT_INCLUDE_DIR NAMES libgccjit.h
        PATHS
        /usr/lib/gcc/x86_64-linux-gnu/12/include
        /usr/include
)

find_library(LIBGCCJIT_LIBRARY NAMES gccjit
        PATHS
        /usr/lib/gcc/x86_64-linux-gnu/12
        /usr/lib/x86_64-linux-gnu
        /usr/lib
)

add_subdirectory(submodules/antlr4/runtime/Cpp/)
include_directories(submodules/antlr4/runtime/Cpp/runtime/src)
include_directories(src/antlr4/)

# Gestisce i messaggi di errore se la libreria non viene trovata
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(LibGccJit
        REQUIRED_VARS LIBGCCJIT_LIBRARY LIBGCCJIT_INCLUDE_DIR
)

include_directories(${LIBGCCJIT_INCLUDE_DIR})

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

include_directories(/home/gyankos/Scaricati/libtorch-shared-with-deps-2.14.0+cu132/libtorch/include)
include_directories(include)
include_directories(submodules/catch2/src)
include_directories(submodules/indicators/include)
add_subdirectory(submodules/catch2)
add_subdirectory(submodules/nanobind)
include_directories(submodules/Mach7/code)

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

add_executable(distributed_gradient_descent examples/distributed_gradient_descent.cpp)
target_link_libraries(distributed_gradient_descent "${TORCH_LIBRARIES}")
apply_maximum_cxx_standard(distributed_gradient_descent)

## --- Modulo Nanobind ---
#nanobind_add_module(metatensor_core NOMINSIZE src/cpp/logds/metatensor/bindings.cpp)
#target_link_libraries(metatensor_core PRIVATE ${TORCH_LIBRARIES})
## RISOLUTIVO PER BAD_CAST: Forziamo la visibilità pubblica dei metatipi sui binari emessi
#set_target_properties(metatensor_core PROPERTIES
#        CXX_VISIBILITY_PRESET default
#        VISIBILITY_INLINES_HIDDEN OFF
#)
#apply_maximum_cxx_standard(metatensor_core) # Forzerà la modalità C++2c/26 abilitando i fold expression
#install(TARGETS metatensor_core LIBRARY DESTINATION .)
add_library(MetaTensorJIT
        src/cpp/logds/tensormetadata/AntlrAstBuilderVisitor.cpp
        src/cpp/logds/tensormetadata/JitShapeInferenceEngine.cpp
        src/cpp/logds/tensormetadata/Mach7ShapeInferenceBridge.cpp
        src/cpp/logds/tensormetadata/JitFlyweightDecorator.cpp
        include/logds/tensormetadata/JitFlyweightDecorator.h
        src/cpp/logds/libtorchwrapper/CreateMethod.cpp
        include/logds/libtorchwrapper/CreateMethod.h
        src/cpp/logds/libtorchwrapper/CreateLibTorchModule.cpp
        include/logds/libtorchwrapper/CreateLibTorchModule.h
        src/antlr4/TensorLangBaseListener.cpp
        src/antlr4/TensorLangBaseVisitor.cpp
        src/antlr4/TensorLangLexer.cpp
        src/antlr4/TensorLangListener.cpp
        src/antlr4/TensorLangParser.cpp
        src/antlr4/TensorLangVisitor.cpp
)
apply_maximum_cxx_standard(MetaTensorJIT)
target_link_libraries(MetaTensorJIT )
target_link_libraries(MetaTensorJIT  "${TORCH_LIBRARIES}" antlr4_shared )

add_executable(GirlCalledAlex make_module.cpp)
apply_maximum_cxx_standard(GirlCalledAlex)
target_link_libraries(GirlCalledAlex  "${TORCH_LIBRARIES}" antlr4_shared MetaTensorJIT)