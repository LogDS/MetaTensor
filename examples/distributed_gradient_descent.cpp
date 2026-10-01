/*
* This file is part of the MetaTensor distribution (https://github.com/logds/MetaTensor).
 * Copyright (c) 2026 Giacomo Bergami, PhD
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 */

#include <indicators/progress_bar.hpp>
#include <indicators/cursor_control.hpp>
#include <torch/torch.h>
#include <logds/metatensor/MetaTensor.h>
#include <logds/metatensor/GradientTape.h>
#include <logds/metatensor/DistributedContext.h>
#include <logds/metatensor/Init.h>
#include <iostream>
#include <cmath>
#include <string>
#include <iomanip>
#include <sstream>

int main() {
    // 1. DYNAMIC HARDWARE DETECTION
    // Automatically figures out if it's running via mpirun/mpiexec or standalone
    DistributedContext::init();

    // UI Guard: Only the root node (or single-machine process) updates the terminal UI
    bool show_ui = DistributedContext::is_root();
    if (show_ui) {
        indicators::show_console_cursor(false);
        std::cout << "=== Active Context-Driven Dual Execution Runtime (C++26) ===\n";
    }

    float learning_rate = 0.05f;       // Incrementato per accelerare i passi di Adam
    constexpr int max_epochs = 250;    // Aumentato per dare il tempo al grafo di convergere
    constexpr float convergence_threshold = 1e-3f;

    auto device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;

    // examples/distributed_gradient_descent.cpp

    // 1. Inizializziamo i tensori globali vuoti/zero su tutti i nodi
    DMetaTensor<float, 128, 64> X_global(device, InitPattern::Zeros);
    DMetaTensor<float, 64, 1>   W_true(device, InitPattern::Zeros);

    // 2. Solo il nodo ROOT (Rank 0) genera i reali dati casuali del problema
    if (DistributedContext::is_root()) {
        X_global.storage = torch::rand({128, 64}, torch::TensorOptions().device(device));
        W_true.storage = torch::randn({64, 1}, torch::TensorOptions().device(device));
    }

    // 3. RISOLUTIVO: Sincronizziamo i dati iniziali su tutto il cluster!
    // Sfruttiamo l'AllReduce funzionale che restituisce lo stesso valore su tutti i nodi.
    auto X_sync = X_global.distributed_allreduce_sum();
    auto W_true_sync = W_true.distributed_allreduce_sum();

    // Generazione coerente e biunivoca dei target Y_true su ciascun nodo del cluster
    auto Y_true_global = (X_sync * W_true_sync).element_wise_sigmoid();

    // 4. DATA PARALLEL SCATTER: Ora lo scatter lavora su matrici identiche e coordinate!
    auto X_local = X_sync.template distributed_scatter<0>();
    auto Y_true_local = Y_true_global.template distributed_scatter<0>();


    // MODEL PARAMETERS (Replicated and initialized identically on all nodes)
    DMetaTensor<float, 64, 1> W(device, InitPattern::Zeros);

    // 4. BIND MODEL STRUCTURE TO THE GRADIENT RECORDING TAPE
    GradientTape tape(W);
    tape.set_optimizer(OptimizerType::Adam);
    tape.set_lr_decay(DecayType::Exponential, 0.95f); // 5% learning rate decay per epoch

    bool early_stopping_triggered = false;

    // Allocate the progress metrics block exclusively for the UI controller process
    std::unique_ptr<indicators::ProgressBar> bar;
    if (show_ui) {
        bar = std::make_unique<indicators::ProgressBar>(
            indicators::option::BarWidth{50}, indicators::option::Start{"["},
            indicators::option::Fill{"="}, indicators::option::Lead{">"},
            indicators::option::Remainder{" "}, indicators::option::End{"]"},
            indicators::option::PostfixText{"Initializing..."},
            indicators::option::ForegroundColor{indicators::Color::green},
            indicators::option::ShowPercentage{true}, indicators::option::ShowElapsedTime{true},
            indicators::option::ShowRemainingTime{true},
            indicators::option::FontStyles{std::vector<indicators::FontStyle>{indicators::FontStyle::bold}},
            indicators::option::MaxProgress{max_epochs}
        );
    }

    // 5. EMBEDDED CONTEXT-DRIVEN OPTIMIZATION ENVIRONMENT
    for (int epoch = 1; epoch <= max_epochs; ++epoch) {
        float host_loss_value = 0.0f;

        // RAII Context Block: Hooks backward pass & mathematical optimizer updates automatically at the closing brace
        if (auto epoch_context = tape.next_epoch(learning_rate, early_stopping_triggered)) {

            // Process forward nodes over local chunks (32 items if distributed, 128 items if local)
            auto Y_pred_local = (X_local * W).element_wise_sigmoid();
            auto error_local = Y_pred_local - Y_true_local;
            auto square_error_local = error_local.element_wise_mul(error_local);

            // Multi-axis reduction collapsing partial shapes down to a unit 0-D loss scalar wrapper
            auto loss = square_error_local.reduce_all_sum();
            host_loss_value = loss;

            // Feed the computational context to evaluate structural integrity checks
            epoch_context.feed_loss(loss);

            if (show_ui) {
                std::stringstream ss;
                ss << "Epoch " << epoch << "/" << max_epochs << " | Loss: " << std::fixed << std::setprecision(6) << host_loss_value;
                bar->set_option(indicators::option::PostfixText{ss.str()});
            }

        } // <--- SCOPE CLOSES SLYLY HERE!
          // Multi-Node: Evaluates loss.backward() -> collective AllReduce SUM over cluster network topology
          // -> scales by 1/N -> modifies W weights -> triggers CUDACachingAllocator::emptyCache().
          // Single-Machine: Evaluates loss.backward() -> bypasses AllReduce (No-Op) -> modifies W weights local context.
        else {
            if (show_ui) bar->set_option(indicators::option::PostfixText{"[FAILED] Early Stop Triggered!"});
            break;
        }

        if (show_ui) bar->tick();

        // Break early if global converged target threshold guarantees mathematical convergence
        if (host_loss_value < convergence_threshold) {
            if (show_ui) {
                std::stringstream ss;
                ss << "[CONVERGED] Target hit at epoch " << epoch << " | Loss: " << host_loss_value;
                bar->set_option(indicators::option::PostfixText{ss.str()});
            }
            break;
        }
    }

    if (show_ui) {
        bar->mark_as_completed();
        indicators::show_console_cursor(true);
    }

    // Surgical memory reclamation
    W.clear(); X_global.clear(); Y_true_global.clear(); W_true.clear(); X_local.clear(); Y_true_local.clear();

    if (show_ui) {
        std::cout << "\nTraining execution loop terminated. Memory contexts unmapped gracefully.\n";
    }

    return 0;
}
