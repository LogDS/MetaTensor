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
#ifndef TENSORLIBRARY_GRADIENTTAPE_H
#define TENSORLIBRARY_GRADIENTTAPE_H

#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <tuple>

// =============================================================================
// GRADIENT TAPE MULTI-TENSORE AVANZATO (C++26)
// =============================================================================
struct GradientTape {
    std::vector<torch::Tensor*> watched_storages;

    // Costruttore Variadic: Registra e attiva automaticamente il watch sui tensori inseriti
    template <typename... TensorTypes>
    GradientTape(TensorTypes&... tensors) {
        torch::autograd::GradMode::set_enabled(true);
        // Espansione del pacchetto tramite fold expression per marcare i tensori e salvare i puntatori
        ([&]() {
            tensors.watch();
            watched_storages.push_back(&tensors.storage);
        }(), ...);
    }

    // Distruttore RAII: Esegue l'unwatch automatico alla chiusura dello scope del nastro
    ~GradientTape() {
        for (auto* storage_ptr : watched_storages) {
            if (storage_ptr && storage_ptr->defined()) {
                storage_ptr->set_requires_grad(false);
            }
        }
    }

    // Calcola il backward ed estrae i gradienti accumulati in una tupla statica
    template <typename LossT, typename... TensorTypes>
    auto gradients(LossT& loss_tensor, const TensorTypes&... tensors) {
        static_assert(LossT::Rank == 0, "[ERR_GRAD] Il gradiente richiede una Loss scalare 0-D.");

        loss_tensor.storage.backward();
        return std::tuple<TensorTypes...>{ tensors.grad()... };
    }
};

#endif //TENSORLIBRARY_GRADIENTTAPE_H
