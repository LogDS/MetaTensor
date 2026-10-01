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

#ifndef TENSORLIBRARY_RELATIONALEINSUMCOMPILER_H
#define TENSORLIBRARY_RELATIONALEINSUMCOMPILER_H

#include <torch/torch.h>
#include <array>
#include <iostream>
#include <utility>

// 1. Tag per identificare l'origine degli assi nella proiezione relazionale
enum class Source { Left, Right };

template <Source Src, size_t Index>
struct Axis {
    static constexpr Source source = Src;
    static constexpr size_t index = Index;
};

// Alias espressivi per le proiezioni
template <size_t Index> using L = Axis<Source::Left, Index>;
template <size_t Index> using R = Axis<Source::Right, Index>;

// =============================================================================
// COMPILATORE RELAZIONALE CONSTEXPR (Generatore di stringhe HLO/Einsum)
// =============================================================================
template <size_t RankA, size_t RankB, typename... Projections>
struct RelationalEinsumCompiler {
    static constexpr size_t total_size = RankA + 1 + RankB + 2 + sizeof...(Projections) + 1;

    static constexpr auto generate() {
        std::array<char, total_size> buffer{};
        size_t ptr = 0;

        // Mappatura alfabeto del tensore Sinistro (L) -> 'a', 'b', 'c'...
        for (size_t i = 0; i < RankA; ++i) buffer[ptr++] = static_cast<char>('a' + i);
        buffer[ptr++] = ',';

        // Mappatura alfabeto del tensore Destro (R) -> 'd', 'e'...
        size_t base_R = RankA;
        for (size_t i = 0; i < RankB; ++i) buffer[ptr++] = static_cast<char>('a' + base_R + i);

        // Identificazione automatica dell'asse contratto implicito
        constexpr std::array<Source, sizeof...(Projections)> out_sources = { Projections::source... };
        constexpr std::array<size_t, sizeof...(Projections)> out_indices = { Projections::index... };

        size_t unprojected_L = 999;
        size_t unprojected_R = 999;

        for (size_t i = 0; i < RankA; ++i) {
            bool found = false;
            for (size_t j = 0; j < out_sources.size(); ++j) {
                if (out_sources[j] == Source::Left && out_indices[j] == i) found = true;
            }
            if (!found) unprojected_L = i;
        }

        for (size_t i = 0; i < RankB; ++i) {
            bool found = false;
            for (size_t j = 0; j < out_sources.size(); ++j) {
                if (out_sources[j] == Source::Right && out_indices[j] == i) found = true;
            }
            if (!found) unprojected_R = i;
        }

        // Se un asse viene contratto, unifichiamo i caratteri alfabetici
        if (unprojected_L != 999 && unprojected_R != 999) {
            size_t target_pos = RankA + 1 + unprojected_R;
            buffer[target_pos] = static_cast<char>('a' + unprojected_L);
        }

        buffer[ptr++] = '-';
        buffer[ptr++] = '>';

        // Generazione stringa finale di Output basata sulle proiezioni
        auto write_out = [&](Source src, size_t index) {
            if (src == Source::Left) {
                buffer[ptr++] = static_cast<char>('a' + index);
            } else {
                if (index == unprojected_R && unprojected_L != 999) {
                    buffer[ptr++] = static_cast<char>('a' + unprojected_L);
                } else {
                    buffer[ptr++] = static_cast<char>('a' + RankA + index);
                }
            }
        };
        (write_out(Projections::source, Projections::index), ...);
        buffer[ptr++] = '\0';

        return std::pair{buffer, std::pair{unprojected_L, unprojected_R}};
    }

    static constexpr auto meta_data = generate();
    static constexpr auto string_storage = meta_data.first;
    static constexpr size_t contracting_axis_L = meta_data.second.first;
    static constexpr size_t contracting_axis_R = meta_data.second.second;
};


#endif //TENSORLIBRARY_RELATIONALEINSUMCOMPILER_H
