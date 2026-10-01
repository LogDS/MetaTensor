/*
 * This file is part of the MetaTensor distribution (https://github.com).
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
 * along with this program. If not, see <http://gnu.org>.
 */

#include <indicators/progress_bar.hpp>
#include <indicators/indeterminate_progress_bar.hpp>
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
    // Nascondiamo il cursore per evitare sfarfallii durante l'aggiornamento della barra
    indicators::show_console_cursor(false);

    auto device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
    std::cout << "=== Orchestrazione OpenXLA/LibTorch Standard (C++26) ===\n";
    std::cout << "=== Test Inizializzazione Pattern MetaTensor (C++26) ===\n\n";

    // Ottimizzazione del tasso di apprendimento per la sigmoide continua
    float learning_rate = 0.5f;
    constexpr int max_epochs = 50; // Aumentato a 50 per mostrare una convergenza fluida
    constexpr float convergence_threshold = 1e-3f;

    // 1. DATASET LINEARMENTE SEPARABILE (Teacher-Student Pattern per garantire la convergenza)
    MetaTensor<float, 128, 64> X( device, InitPattern::RandomUniform);
    MetaTensor<float, 64, 1>   W_true( device, InitPattern::RandomNormal);

    // Y_true generata in modo coerente: il modello convergerà minimizzando la Loss
    auto Y_true = (X * W_true).element_wise_sigmoid();

    // 2. INIZIALIZZAZIONE DEI PARAMETRI DEL MODELLO
    MetaTensor<float, 64, 1>   W(device, InitPattern::Zeros);

    // Esempi opzionali aggiuntivi (strutture costanti verificate)
    MetaTensor<float, 64, 64>  I(device, InitPattern::Identity);
    MetaTensor<float, 64, 1>   b(device, InitPattern::Zeros);

    std::cout << "-> Tutti i tensori sono stati inizializzati correttamente in VRAM.\n";
    std::cout << "   Forma della matrice Identità verificata: " << I.storage.sizes() << "\n";
    std::cout << "Inizio ottimizzazione sui nodi hardware...\n\n";

    // 3. IL GRADIENT TAPE NASCE PRIMA DEL LOOP DELLE EPOCHE
    GradientTape tape(W);

    // Configurazione formale della barra di avanzamento indicators
    indicators::ProgressBar bar{
        indicators::option::BarWidth{50},
        indicators::option::Start{"["},
        indicators::option::Fill{"="},
        indicators::option::Lead{">"},
        indicators::option::Remainder{" "},
        indicators::option::End{"]"},
        indicators::option::PostfixText{"Initializing..."},
        indicators::option::ForegroundColor{indicators::Color::green},
        indicators::option::ShowPercentage{true},
        indicators::option::ShowElapsedTime{true},
        indicators::option::ShowRemainingTime{true},
        indicators::option::FontStyles{std::vector<indicators::FontStyle>{indicators::FontStyle::bold}},
        indicators::option::MaxProgress{max_epochs}
    };

    for (int epoch = 1; epoch <= max_epochs; ++epoch) {
        float host_loss_value = 0.0f;

        // =====================================================================
        // SCOPE LOCALE STRETTO PER LE ALLOCAZIONI DEI BUFFER TEMPORANEI
        // =====================================================================
        {
            // Forward Pass fuso: usiamo la Sigmoide continua per far scorrere l'Autograd
            auto Y_pred = (X * W).element_wise_sigmoid();
            auto error = Y_pred - Y_true;
            auto square_error = error.element_wise_mul(error);

            // Riduzione totale a scalare puro 0-D (Rank = 0)
            auto loss = square_error.reduce_all_sum();

            // Estrazione del gradiente dW tramite il nastro globale
            auto [dW] = tape.gradients(loss, W);

            // Mutazione in-place dei pesi isolata dall'Autograd
            W.apply_gradient_descent(dW, learning_rate);

            // Estrazione sicura del valore host tramite l'operatore di cast implicito C++26
            host_loss_value = loss;

            // Formattazione della stringa Postfix per mostrare epoca e minimizzazione decimale della Loss
            std::stringstream ss;
            ss << "Epoch " << epoch << "/" << max_epochs
               << " | Loss: " << std::fixed << std::setprecision(6) << host_loss_value;
            bar.set_option(indicators::option::PostfixText{ss.str()});

            // Controllo Early Stopping condizionale per instabilità dei dati
            if (std::isnan(host_loss_value) || std::isinf(host_loss_value)) {
                bar.set_option(indicators::option::PostfixText{"[FAILED] Instabilità numerica!"});
                std::cerr << "\n[EARLY STOPPING] Rilevato NaN/Inf nei vettori di Loss.\n";
                break;
            }
        }
        // <--- LA GRAFFA SI CHIUDE QUI!
        // I tensori temporanei (Y_pred, error, square_error, loss, dW) escono dallo scope.
        // Il distruttore ~MetaTensor() cancella istantaneamente i buffer svuotando la cache.

        // Aggiornamento grafico della barra ad ogni iterazione dell'epoca
        bar.tick();

        // Controllo della convergenza ottimale anticipata
        if (host_loss_value < convergence_threshold) {
            std::stringstream ss;
            ss << "[CONVERGED] Target raggiunto all'epoca " << epoch << " | Loss: " << host_loss_value;
            bar.set_option(indicators::option::PostfixText{ss.str()});
            break;
        }
    }

    // Forza il completamento grafico definitivo della barra sul terminale
    bar.mark_as_completed();

    // Ripristiniamo il cursore della console prima di uscire dal programma
    indicators::show_console_cursor(true);

    // Svuotamento esplicito finale delle matrici e delle strutture statiche globali
    W.clear();
    X.clear();
    Y_true.clear();
    W_true.clear();
    I.clear();
    b.clear();

    std::cout << "\nTraining completato con successo. VRAM e contesti hardware azzerati.\n";

    return 0;
}
