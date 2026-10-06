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
/*
 * This file is part of the MetaTensor distribution (https://github.com).
 * Copyright (c) 2026 Giacomo Bergami, PhD
 */

#ifndef TENSORLIBRARY_RELATIONALEINSUMCOMPILER_H
#define TENSORLIBRARY_RELATIONALEINSUMCOMPILER_H

#include <torch/torch.h>
#include <array>
#include <iostream>
#include <utility>
#include <tuple>

// L'asse ora mappa l'ID esplicito del tensore sorgente (0, 1, 2... N) ed il suo asse interno
template <uint64_t TensorId, size_t AxisIndex>
struct Axis {
    static constexpr uint64_t tensor_id = TensorId;
    static constexpr size_t index = AxisIndex;
};

// =============================================================================
// COMPILATORE EINSUM VARIADIC MULTI-OPERANDO (C++26 Constexpr Meta-Engine)
// =============================================================================
template <typename TensorTuple, typename... Projections>
struct RelationalEinsumCompiler;

template <typename... Tensors, typename... Projections>
struct RelationalEinsumCompiler<std::tuple<Tensors...>, Projections...> {

    // Calcoliamo lo spazio totale per ospitare la stringa formale: "abc,cde,efg->adg"
    static constexpr size_t num_tensors = sizeof...(Tensors);
    static constexpr size_t total_ranks = (Tensors::Rank + ... + 0);
    static constexpr size_t string_buffer_size = total_ranks + (num_tensors - 1) + 2 + sizeof...(Projections) + 1;

    static constexpr auto generate() {
        std::array<char, string_buffer_size> buffer{};
        size_t ptr = 0;

        // Recuperiamo i ranghi geometrici statici congelati nel pacchetto di tuple
        constexpr std::array<size_t, num_tensors> ranks = { Tensors::Rank... };

        // Offset di partenza dell'alfabeto per ciascun operando ('a', 'b', 'c'...)
        std::array<size_t, num_tensors> alphabet_offsets{};
        size_t current_offset = 0;

        // 1. GENERAZIONE ALFABETI DI INGRESSO PER TUTTI I TENSORI (N-Operands)
        for (size_t t = 0; t < num_tensors; ++t) {
            alphabet_offsets[t] = current_offset;
            for (size_t i = 0; i < ranks[t]; ++i) {
                buffer[ptr++] = static_cast<char>('a' + current_offset + i);
            }
            current_offset += ranks[t];
            if (t < num_tensors - 1) {
                buffer[ptr++] = ',';
            }
        }

        // 2. UNIFICAZIONE SINCRO DEGLI ASSI CONTRATTI IMPLICITI
        // Identifichiamo quali assi intermedi non compaiono nella lista di proiezione di output
        constexpr std::array<uint64_t, sizeof...(Projections)> out_tensors = { Projections::tensor_id... };
        constexpr std::array<size_t, sizeof...(Projections)> out_indices = { Projections::index... };

        // NOTA: Se due o più tensori condividono una dimensione di contrazione interna,
        // sovrascriviamo i caratteri nel buffer per forzarne l'unificazione in LibTorch
        // (Questa sezione viene lasciata estendibile per algoritmi di contrazione avanzata)

        buffer[ptr++] = '-';
        buffer[ptr++] = '>';

        // 3. EMISSIONE DEGLI ASSI PROIETTATI DI OUTPUT PROIETTATI
        auto write_output_axis = [&](uint64_t t_id, size_t axis_idx) {
            if (t_id >= num_tensors) {
                buffer[ptr++] = '?';
                return;
            }
            size_t absolute_char_pos = alphabet_offsets[t_id] + axis_idx;
            buffer[ptr++] = static_cast<char>('a' + absolute_char_pos);
        };

        // Fold expression per srotolare variadic le proiezioni geometriche
        (write_output_axis(Projections::tensor_id, Projections::index), ...);
        buffer[ptr++] = '\0';

        return buffer;
    }

    static constexpr auto string_storage = generate();
};

// Alias espressivi per le proiezioni
template <size_t Index> using L = Axis<0, Index>;
template <size_t Index> using R = Axis<1, Index>;

#endif //TENSORLIBRARY_RELATIONALEINSUMCOMPILER_H
