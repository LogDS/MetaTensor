# MetaTensor (C++26 Edition)

[![License: GPL v3](https://shields.io)](https://gnu.org)
[![Language: C++26](https://shields.io)](https://cppreference.com)
[![Backend: LibTorch / OpenXLA](https://shields.io)](https://pytorch.org)

**MetaTensor** is a strongly-typed, compile-time verified algebraic-relational tensor engine written in **C++26** and built on top of the bare-metal infrastructures of **OpenXLA (StableHLO)** and **LibTorch (ATen Core)**.

The framework shifts tensor geometric validation (rank verification, axis alignment, and dimensions matching) and physical layout semantics entirely from runtime tracking to **compile-time** via Non-Type Template Parameters (NTTP) and C++20/26 Constraints (`requires`). This enables aggressive graph-kernel fusion inside the OpenXLA/StableHLO backend while enforcing a surgical, deterministic hardware memory reclamation protocol (**Zero-Caching**) to prevent deferred allocations or memory leaks inside GPU cluster nodes.

---

## 🚀 Key Features

*   **Compile-Time Structural Verification:** Rank, axis bounds, and physical layout parameters are permanently bound to the class type signatures (`MetaTensor<T, Layout, Dims...>`). Any geometric mismatch, illegal broadcasting, or unmappable sparse-dense intersection triggers a blocking compiler error (`static_assert`) instead of a runtime crash.
*   **Monomorphized Storage Layouts:** Native support for both dense matrices and high-dimensional coordinate sparse tensors (`StorageLayout::SparseCOO`). The engine utilizes compile-time branching (`if constexpr`) to automatically choose between ultra-fast dense BLAS operations, native sparse operations (`torch::mm`), or transient on-the-fly auto-densifications.
*   **Context-Driven Active Epoches:** Replaces passive autograd loops with declarative, RAII-enforced active epoch contexts (`tape.next_epoch()`). Optimization steps, backpropagation, and hardware cache evacuations are fully automated and synchronized at the closing brace of conditional block scopes (`if`).
*   **Advanced Parameter Optimization:** Integrated variadic optimization states supporting **SGD with Momentum** and **Adam** alongside configurable **Learning Rate Decay Schemes** (Step and Exponential Decay) tracked directly inside the session tape.
*   **Functional Generalized Operators:** Multi-axis existential quantification ($\exists$) governed by arbitrary lambda predicates and relational aggregations (such as associative MapReduce contraction steps) resolved entirely at compile-time.
*   **Dual Local/Distributed Parallelism:** Fully integrated data parallelism that dynamically detects whether it is running via an MPI process manager (`mpirun`/`mpiexec`) or on a standalone workstation, providing a zero-overhead fallback to local hardware.

---

## 📂 Core API Architecture

The engine is engineered as a zero-overhead, modular abstraction layer:

### 1. Storage Layouts & Pattern-Driven Inits (`Init.h` & `StorageLayout.h`)
```cpp
enum class StorageLayout { Dense, SparseCOO };

enum class InitPattern {
    RandomNormal,   // Gaussian normal distribution sampling
    RandomUniform,  // Uniform distribution sampling over [0, 1)
    Zeros,          // Fills structure with 0.0f (e.g., Bias Tensors)
    OnOnes,         // Fills structure with 1.0f (e.g., Constant Targets)
    Identity        // Identity Matrix (Requires rank == 2 and square dimensions)
};
```

### 2. Core Tensor Wrapper Definition (`MetaTensor.h`)
```cpp
template <typename T, StorageLayout Layout, size_t... Dims>
class MetaTensor {
public:
    static constexpr size_t Rank = sizeof...(Dims);
    static constexpr std::array<size_t, Rank> Shape = { Dims... };
    static constexpr StorageLayout layout = Layout;
    using value_type = T;

    torch::Tensor storage;

    // Bare-Metal and Structured Constructors
    MetaTensor(torch::Device device = torch::kCPU, InitPattern pattern = InitPattern::RandomNormal);
    template <typename TupleT>
    MetaTensor(const std::vector<TupleT>& entries, const std::vector<T>& values, torch::Device device = torch::kCPU);
    MetaTensor(torch::Tensor t);
    
    static auto from_scalar(T value, torch::Device device = torch::kCPU);
    auto to_dense() const;
    void clear();
    ~MetaTensor();

    // Relational Tensor Algebra & Multi-Axis Existential Quantification (∃)
    template <CellOp Op> auto apply() const;
    auto element_wise_sigmoid() const;
    template <size_t... Perm> auto permute_axes() const;
    template <size_t Axis> auto reduce_sum() const;
    auto reduce_all_sum() const; 
    
    template <size_t... ReduceAxes, typename PredicateLambda>
    auto evaluate_existential(PredicateLambda&& predicate) const;
    template <size_t... RetainedAxes> 
    auto aggregate(AggregationOp op) const;

    // Type-Safe Multi-Dimensional Cell Extraction (Enforces N == Rank to prevent illegal slicing)
    template <size_t N> requires (N == Rank)
    TensorCellProxy operator[](const std::array<size_t, N>& coords);

    // Polymorphic Operator Overloading (Auto-Layout Dispatching)
    template <StorageLayout RLayout, size_t... RDims> auto operator+(const MetaTensor<T, RLayout, RDims...>& other) const;
    template <StorageLayout RLayout, size_t... RDims> auto operator*(const MetaTensor<T, RLayout, RDims...>& other) const;
    template <StorageLayout RLayout, size_t... RDims> auto operator%(const MetaTensor<T, RLayout, RDims...>& other) const; // 3D Cross Product

    // Dual-Execution Data Parallel Distributed Operations
    template <size_t PartitionAxis = 0> auto distributed_scatter() const;
    auto distributed_allreduce_sum() const;

    // C++26 Safe Scalar Primitive Casting (Enabled only if Rank == 0 or dimensions collapse to 1)
    template <typename U = T> requires (Rank == 0 || (... && (Dims == 1))) operator U() const;
};
```

### 3. Active Optimization Gradient Tape (`GradientTape.h`)
```cpp
enum class OptimizerType { SGD, Momentum, Adam };
enum class DecayType { None, Step, Exponential };

struct GradientTape {
    OptimizerState optimizer_state;

    template <typename... TensorTypes> GradientTape(TensorTypes&... tensors);
    void set_optimizer(OptimizerType type);
    void set_lr_decay(DecayType scheme, float gamma, int64_t steps = 10);

    // Active Epoch Context Generator (Binds parameter references and eliminates manual loops)
    template <typename... TensorTypes>
    auto next_epoch(float lr, bool& early_stop, TensorTypes&... tensors);
};
```

---

## 🛠️ Hybrid Dual-Execution Training Example

This script demonstrates a real optimization step under an active **Adam** optimizer with **Exponential Learning Rate Decay**. If executed normally (`./main`), it runs as a local single-machine process over the full dataset. If spawned via `mpirun -n 4 ./main`, it automatically shards the data over 4 nodes, processes partial batch steps, and synchronizes network weights over GLOO/NCCL ring topologies.

```cpp
#include <indicators/progress_bar.hpp>
#include <indicators/cursor_control.hpp>
#include <torch/torch.h>
#include <logds/metatensor/MetaTensor.h>
#include <logds/metatensor/GradientTape.h>
#include <logds/metatensor/DistributedContext.h>
#include <logds/metatensor/Init.h>
#include <iostream>
#include <iomanip>

int main() {
    // 1. DYNAMIC DISTRIBUTED CLUSTER CONFIGURATION
    DistributedContext::init();
    bool show_ui = DistributedContext::is_root();
    if (show_ui) indicators::show_console_cursor(false);

    float learning_rate = 0.05f; 
    constexpr int max_epochs = 250;
    constexpr float convergence_threshold = 1e-3f;
    auto device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;

    // 2. COORDINATED DATASET INITIALIZATION
    DMetaTensor<float, 128, 64> X_global(device, InitPattern::Zeros);
    DMetaTensor<float, 64, 1>   W_true(device, InitPattern::Zeros);

    if (DistributedContext::is_root()) {
        X_global.storage = torch::rand({128, 64}, torch::TensorOptions().device(device));
        W_true.storage = torch::randn({64, 1}, torch::TensorOptions().device(device));
    }

    // Broadcast structures across the network topology via functional reduction
    auto X_sync = X_global.distributed_allreduce_sum();
    auto W_true_sync = W_true.distributed_allreduce_sum();
    auto Y_true_global = (X_sync * W_true_sync).element_wise_sigmoid();

    // 3. DATA PARALLEL SCATTERING WITH AUTOMATIC LOCAL FALLBACK
    auto X_local = X_sync.template distributed_scatter<0>();
    auto Y_true_local = Y_true_global.template distributed_scatter<0>();

    // MODEL PARAMETERS (Synchronized across all cluster workers)
    DMetaTensor<float, 64, 1> W(device, InitPattern::Zeros);

    // 4. OPTIMIZER CONFIGURATION ON THE GRADIENT TAPE
    GradientTape tape(W);
    tape.set_optimizer(OptimizerType::Adam);
    tape.set_lr_decay(DecayType::Exponential, 0.95f); 
    bool early_stopping_triggered = false;

    for (int epoch = 1; epoch <= max_epochs; ++epoch) {
        float host_loss_value = 0.0f;

        // =====================================================================
        // ACTIVE CONTEXT SCOPE BLOCK
        // =====================================================================
        if (auto epoch_context = tape.next_epoch(learning_rate, early_stopping_triggered, W)) {

            // Process forward steps over sharded matrices (32 rows if distributed, 128 if local)
            auto Y_pred_local = (X_local * W).element_wise_sigmoid();
            auto error_local = Y_pred_local - Y_true_local;
            auto loss = error_local.element_wise_mul(error_local).reduce_all_sum();

            host_loss_value = loss;
            // Safe shallow copy ingestion protects the autograd graph against Zero-Caching deallocations
            epoch_context.feed_loss(loss);
            
        } // <--- ACTIVE SCOPE CLOSES DETERMINISTICALLY HERE!
        // Automatically performs loss.backward() -> collective network AllReduce SUM ->
        // computes Adam rolling historical moments -> updates W parameters -> purges VRAM.
    else {
        break;
    }
    if (show_ui) {
            std::cout << "Epoch " << epoch << "/" << max_epochs << " | Synchronized Loss: " << host_loss_value << "\n";
    }
            if (host_loss_value < convergence_threshold) break;
    }
    if (show_ui) indicators::show_console_cursor(true);
    // Hard unmapping of physical device allocations
    W.clear(); X_global.clear(); Y_true_global.clear(); W_true.clear(); X_local.clear(); Y_true_local.clear();
    return 0;
}
```

## 💻 System Requirements & Compilation

Prerequisites

* **Compiler**: A C++ compiler fully compliant with the C++26 standard (GCC 14+, Clang 18+; GCC 12 is supported with -std=gnu++20).
* **Libraries**: LibTorch Toolkit (PyTorch C++ Shared Library) extracted and available on the host path.
* **Distributed Env (Optional)**: OpenMPI or MPICH system runtimes installed on the node cluster.
* 
## 📜 License
This project is licensed under the terms of the GNU General Public License v3 (GPLv3). See the LICENSE file for detailed provisions.

## 👥 Authors and Contacts
 *  Author: Giacomo Bergami, PhD
 * Repository: github.com/logds/metatensor