# MetaTensor (C++26 Edition)

[![License: GPL v3](https://shields.io)](https://gnu.org)
[![Language: C++26](https://shields.io)](https://cppreference.com)
[![Backend: LibTorch / OpenXLA](https://shields.io)](https://pytorch.org)

**MetaTensor** is a strongly-typed, compile-time verified algebraic-relational tensor engine written in **C++26** and built on top of the bare-metal infrastructures of **OpenXLA (StableHLO)** and **LibTorch (ATen Core)**.

The framework shifts tensor geometric validation (rank verification, axis alignment, and dimensions matching) entirely from runtime execution to **compile-time** via Non-Type Template Parameters (NTTP) and C++20/26 Constraints (`requires`). This allows the underlying compiler to aggressively fuse computational graphs and execute a surgical hardware memory reclamation protocol (**Zero-Caching**) to prevent any deferred allocations or memory leaks inside GPU cluster nodes.

---

## 🚀 Key Features

*   **Compile-Time Structural Verification:** Rank and axis bounds are permanently bound to the class type signatures (`MetaTensor<T, Dims...>`). Any geometric mismatch or illegal broadcasting triggers a blocking compiler error (`static_assert`) instead of a catastrophic runtime segmentation fault or exception.
*   **Relational Axis Projections:** Multi-dimensional contractions and *Einstein Sums* (`einsum`) completely bypass slow runtime string parsing. Instead, they are declared using pure relational coordinate projections (`L<0>, R<1>`). The C++ compiler deduces the contracting axis and generates the optimal HLO string representation behind the scenes at zero runtime cost.
*   **Deterministic RAII Memory Recovery:** The destructor of the tensor wrapper bypasses PyTorch's lazy caching allocator pool by directly invoking `c10::cuda::CUDACachingAllocator::emptyCache()`. The moment an intermediate temporary tensor exits its local block scope `{}`, its VRAM allocation is physically unmapped at the hardware driver level.
*   **C++26 Implicit Scalar Casts:** Tensors that collapse to unit dimensions (such as a 0-D global loss scalar) support a type-safe implicit conversion operator to native C++ primitive types (`float`, `double`), completely removing the need for boilerplate `.item()` extraction methods at runtime.

---

## 📂 Core API Architecture

The engine is engineered as a zero-overhead, modular abstraction layer:

### 1. Pattern-Driven Initialization (`Init.h`)
Differentiates VRAM vector allocation routines using static structural flags, validating geometric restrictions (e.g., identity matrices) at compile-time.
```cpp
enum class InitPattern {
    RandomNormal,   // Gaussian normal distribution sampling
    RandomUniform,  // Uniform distribution sampling over [0, 1)
    Zeros,          // Fills structure with 0.0f (e.g., Bias Tensors)
    OnOnes,         // Fills structure with 1.0f (e.g., Constant Targets)
    Identity        // Identity Matrix (tf.eye - Enforces static_assert M == N)
};
```

### 2. Core Tensor Wrapper Definition (`MetaTensor.h`)
```cpp
template <typename T, size_t... Dims>
class MetaTensor {
public:
    static constexpr size_t Rank = sizeof...(Dims);
    static constexpr std::array<size_t, Rank> Shape = { Dims... };
    torch::Tensor storage;

    // Bare-Metal Constructors & RAII Cleanups
    MetaTensor(torch::Device device = torch::kCPU, InitPattern pattern);
    MetaTensor(torch::Tensor t);
    void clear();
    ~MetaTensor();

    // Relational Tensor Algebra Framework (The 12 Mathematized Operations)
    auto element_wise_sigmoid() const;
    template <size_t... Perm> auto permute_axes() const;
    template <size_t Axis> auto reduce_sum() const;
    auto reduce_all_sum() const; // Collapses structure to a pure 0-D scalar
    template <size_t NumSegments, typename IdTensorT> auto segment_sum(const IdTensorT& segment_ids) const;
    template <size_t Depth> auto one_hot_encoding() const;
    template <typename IndexTensorT> auto gather_nd(const IndexTensorT& indices) const;
    
    // Sections 11 & 12: Relational Theta-Joins and Existential Quantification (∃)
    template <typename RightT> auto tensor_theta_join(const RightT& other) const;
    template <size_t ReduceAxis> auto evaluate_existential_expression() const;

    // Standard C++ Operator Overloading Overriding
    template <size_t... RightDims> auto operator+(const MetaTensor<T, RightDims...>& other) const; // Auto-Broadcasting
    template <size_t... RightDims> auto operator-(const MetaTensor<T, RightDims...>& other) const;
    template <size_t... RightDims> auto operator*(const MetaTensor<T, RightDims...>& other) const; // Universal MatMul via contraction
    template <size_t... RightDims> auto operator%(const MetaTensor<T, RightDims...>& other) const; // Native 3D Cross Product

    // Conditional Masking and Sparse Diagnostics
    template <typename ConditionLambda> auto where(ConditionLambda&& condition, float on=1.f, float off=0.f) const;
    auto clip(float min_val = 0.0f, float max_val = 1.0f) const;
    void pretty_print_sparse(float threshold = 1e-4f) const;
    
    // Hardware Execution Offloading
    auto to_device(torch::Device target_device = torch::kCUDA) const;
    auto to_host() const;

    // C++26 SFINAE / Requires Constraints for Safe Scalar Primitive Casting
    template <typename U = T> requires (Rank == 0 || (... && (Dims == 1))) operator U() const;
};
```

### 3. Multi-Tensor GradientTape Context (`GradientTape.h`)
Encapsulates LibTorch's Autograd mechanism by binding parameter tracking boundaries (`watch` / `unwatch`) directly to the RAII lifespan of the tape, eliminating memory leaks in the computation graph.
```cpp
struct GradientTape {
    // Spawns recording context and triggers explicit parameter watching
    template <typename... TensorTypes> GradientTape(TensorTypes&... tensors);
    
    // Destructor: Automatically revokes parameter flags when tape goes out of scope
    ~GradientTape();

    // Executes a single global backward pass and packs Jacobian matrices into a typed static tuple
    template <typename LossT, typename... TensorTypes>
    auto gradients(LossT& loss_tensor, const TensorTypes&... tensors);
};
```

---

## 🛠️ Production-Grade Training Example: Convergent Loop

This script demonstrates a real optimization step executed on a CUDA GPU accelerator. Dataset blocks persist globally, while all temporary execution parameters are entirely wiped out of the physical VRAM at the end of each epoch using the isolated block scope `{}`.

```cpp
#include <torch/torch.h>
#include <logds/metatensor/MetaTensor.h>
#include <logds/metatensor/GradientTape.h>
#include <logds/metatensor/Init.h>
#include <iostream>

int main() {
    // Automated hardware backend dispatching
    auto device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
    std::cout << "=== Launching Optimization on Hardware Nodes ===\n";

    float learning_rate = 0.5f; 
    int max_epochs = 10;

    // 1. PERSISTENT DATASET: Allocated once outside the loop boundaries
    MetaTensor<float, 128, 64> X(InitPattern::RandomUniform, device);
    MetaTensor<float, 64, 1>   W_true(InitPattern::RandomNormal, device);
    
    // Create stable targets using a Teacher-Student Pattern to guarantee analytical convergence
    auto Y_true = (X * W_true).element_wise_sigmoid();

    // 2. MODEL PARAMETERS TO BE OPTIMIZED (Initialized to Zero)
    MetaTensor<float, 64, 1> W(InitPattern::Zeros, device);

    // 3. GRADIENT TAPE INITIALIZATION BEFORE THE TRAINING LOOP
    // Explicitly binds parameter tracking to the tape's RAII lifecycle context
    GradientTape tape(W);

    for (int epoch = 1; epoch <= max_epochs; ++epoch) {
        float host_loss_value = 0.0f;

        // =====================================================================
        // ISOLATED LOCAL SCOPE FOR TEMPORARY INTERMEDIATE VRAM BUFFERS
        // =====================================================================
        {
            // Fused Forward Pass: Overridden '*' operator automatically invokes relational contraction
            auto Y_pred = (X * W).element_wise_sigmoid();
            
            // Algebraic element-wise operations with compile-time broadcast checks
            auto error = Y_pred - Y_true;
            auto square_error = error.element_wise_mul(error);
            
            // Total multi-axis reduction to a pure 0-D scalar (Rank = 0)
            auto loss = square_error.reduce_all_sum();

            // Direct numerical extraction utilizing the C++26 implicit cast operator
            host_loss_value = loss;

            // Retropropagate gradients back through the HLO graph into a packed static tuple
            auto [dW] = tape.gradients(loss, W);

            // In-place weights mutation isolated from Autograd tracking mechanics
            W.apply_gradient_descent(dW, learning_rate);
            
        } // <--- LOCAL SCOPE EXITS HERE!
          // All temporary tensors (Y_pred, error, square_error, loss, dW) go out of scope.
          // The ~MetaTensor() destructor executes an immediate hardware cache evacuation.
          // Intermediate VRAM footprint is 100% reset before the next iteration begins.

        std::cout << "Epoch " << epoch << " -> Global MSE Loss: " << host_loss_value << "\n";
    }

    // 4. MANUAL CLEANUP OF GLOBAL PERSISTENT STRUCTURES
    W.clear();
    X.clear();
    Y_true.clear();
    W_true.clear();

    return 0;
}
```

---

## 💻 System Requirements & Compilation

### Prerequisites
*   **Compiler:** A C++ compiler fully compliant with the **C++26** standard (GCC 14+, Clang 18+).
* **Dependencies**: LibTorch Toolkit (PyTorch C++ Library) extracted and available on the host machine.

## 📜 License
This project is licensed under the terms of the GNU General Public License v3 (GPLv3). See the LICENSE file for detailed provisions.

## 👥 Authors and Contacts
 *  Author: Giacomo Bergami, PhD
 * Repository: github.com/logds/metatensor