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

#ifndef TENSORLIBRARY_BROADCASTSHAPE_H
#define TENSORLIBRARY_BROADCASTSHAPE_H

#include <string>
#include <cstdint>
#include <array>

// =============================================================================
// HELPER CONSTEXPR PER IL BROADCASTING DELLE SHAPE (Operatore +)
// =============================================================================
template <size_t RankL, size_t RankR>
static constexpr auto deduce_broadcast_shape(const std::array<size_t, RankL>& lhs, const std::array<size_t, RankR>& rhs) {
    constexpr size_t MaxRank = RankL > RankR ? RankL : RankR;
    std::array<size_t, MaxRank> out_shape{};

    for (size_t i = 0; i < MaxRank; ++i) {
        int64_t l_idx = static_cast<int64_t>(RankL) - 1 - i;
        int64_t r_idx = static_cast<int64_t>(RankR) - 1 - i;
        size_t l_dim = (l_idx >= 0) ? lhs[l_idx] : 1;
        size_t r_dim = (r_idx >= 0) ? rhs[r_idx] : 1;

        // Regola formale del broadcasting (stile NumPy/OpenXLA)
        if (l_dim != r_dim && l_dim != 1 && r_dim != 1) {
            // Ritorna una sentinella di errore intercettabile dallo static_assert
            out_shape[0] = 999999; return out_shape;
        }
        out_shape[MaxRank - 1 - i] = (l_dim > r_dim) ? l_dim : r_dim;
    }
    return out_shape;
}

#endif //TENSORLIBRARY_BROADCASTSHAPE_H
