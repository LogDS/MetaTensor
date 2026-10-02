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

#ifndef METATENSOR_TENSORMETADATA_H
#define METATENSOR_TENSORMETADATA_H

#include <string>
#include <variant>
#include <vector>
#include "StorageLayout.h"

typedef enum {
    TYPE_FLOAT,
    TYPE_DOUBLE,
    TYPE_INT32
} DataType;

// Struttura C-Compatible passata a libgccjit++
typedef struct {
    DataType data_type;
    StorageLayout layout;
    size_t rank;            // Se rank == 0, la struttura rappresenta un vero SCALARE atomico
    std::vector<std::variant<std::string,uint64_t>> shape;     // Dimensione fissa 1000 per i metatipi geometrici
    bool is_valid;
} JitTensorMetadata;

#endif //METATENSOR_TENSORMETADATA_H
