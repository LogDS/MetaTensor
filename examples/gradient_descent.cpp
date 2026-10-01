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
#include <logds/metatensor/Init.h>
#include <iostream>
#include <cmath>
#include <string>
#include <iomanip>
#include <sstream>

int main() {
    indicators::show_console_cursor(false);
    auto device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
    std::cout << "=== Orchestrazione Avanzata con Ottimizzatore Adam (C++26) ===\n\n";

    // Nota: Adam richiede un tasso di apprendimento più basso rispetto a SGD puro
    float learning_rate = 0.05f;       // Incrementato per accelerare i passi di Adam
    constexpr int max_epochs = 250;    // Aumentato per dare il tempo al grafo di convergere
    constexpr float convergence_threshold = 1e-3f;

    DMetaTensor<float, 128, 64> X(device, InitPattern::RandomUniform);
    DMetaTensor<float, 64, 1>   W_true(device, InitPattern::RandomNormal);
    auto Y_true = (X * W_true).element_wise_sigmoid();

    DMetaTensor<float, 64, 1>   W(device, InitPattern::Zeros);

    // Inizializzazione del nastro ed attivazione del profilo di ottimizzazione ADAM
    GradientTape tape(W);
    tape.set_optimizer(OptimizerType::Adam);

    // Configura il decadimento: Tipo Esponenziale, riduce del 5% (gamma=0.95) a ogni epoca
    tape.set_lr_decay(DecayType::Exponential, 0.95f);

    bool early_stopping_triggered = false;

    indicators::ProgressBar bar{
        indicators::option::BarWidth{50}, indicators::option::Start{"["},
        indicators::option::Fill{"="}, indicators::option::Lead{">"},
        indicators::option::Remainder{" "}, indicators::option::End{"]"},
        indicators::option::PostfixText{"Initializing..."},
        indicators::option::ForegroundColor{indicators::Color::green},
        indicators::option::ShowPercentage{true}, indicators::option::ShowElapsedTime{true},
        indicators::option::ShowRemainingTime{true},
        indicators::option::FontStyles{std::vector<indicators::FontStyle>{indicators::FontStyle::bold}},
        indicators::option::MaxProgress{max_epochs}
    };

    for (int epoch = 1; epoch <= max_epochs; ++epoch) {
        float host_loss_value = 0.0f;
        if (auto epoch_context = tape.next_epoch(learning_rate, early_stopping_triggered)) {
            auto Y_pred = (X * W).element_wise_sigmoid();
            auto loss = (Y_pred - Y_true).element_wise_mul(Y_pred - Y_true).reduce_all_sum();
            host_loss_value = loss;
            epoch_context.feed_loss(loss);
            std::stringstream ss;
            ss << "Epoch " << epoch << "/" << max_epochs << " | Loss: " << std::fixed << std::setprecision(6) << host_loss_value;
            bar.set_option(indicators::option::PostfixText{ss.str()});
        } else {
            bar.set_option(indicators::option::PostfixText{"[FAILED] Early Stop scattato!"});
            break;
        }
        bar.tick();
    }

    bar.mark_as_completed();
    indicators::show_console_cursor(true);

    W.clear(); X.clear(); Y_true.clear(); W_true.clear();
    std::cout << "\nTraining terminato con successo.\n";

    return 0;
}
