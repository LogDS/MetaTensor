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

#ifndef METATENSOR_ANYMETATENSOR_H
#define METATENSOR_ANYMETATENSOR_H

#include <logds/metatensor/StorageLayout.h>
#include <torch/torch.h>

// =============================================================================
// 1. FORWARD DECLARATION DEL TEMPLATE DERIVATO
// =============================================================================
template <typename T, StorageLayout Layout, size_t... Dims> class MetaTensor;


// =============================================================================
// INTERFACCIA DINAMICA PER RAPPRESENTAZIONI ARBITRARIE A RUNTIME (C++26)
// =============================================================================
class AnyMetaTensor {
public:
    virtual ~AnyMetaTensor() = default;

    // Metodi virtuali puri per interrogare la geometria reale del file a runtime
    virtual size_t get_rank() const = 0;
    virtual std::vector<size_t> get_shape() const = 0;
    virtual StorageLayout get_layout() const = 0;
    virtual torch::Tensor& get_storage() = 0;
    virtual const torch::Tensor& get_storage() const = 0;
    virtual void clear() = 0;

    // Helper per tentare un recupero (downcast) sicuro verso il tipo statico originale
    template <typename T, StorageLayout Layout, size_t... Dims>
    auto* as();
};

#endif //METATENSOR_ANYMETATENSOR_H
