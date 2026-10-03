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

#ifndef METATENSOR_MACH7SHAPEINFERENCEBRIDGE_H
#define METATENSOR_MACH7SHAPEINFERENCEBRIDGE_H


#include <logds/tensormetadata/Metadata.h>
#include <logds/tensormetadata/JitShapeInferenceEngine.h>
#include <optional>
#include <iostream>


class Mach7ShapeInferenceBridge {
public:
    // RISOLUTIVO: Il metodo accetta l'istanza dell'engine per accumulare i vincoli di unificazione contestuale!
    static std::optional<JitTensorMetadata> infer_node_shape(const TensorTyping* node, const JitShapeInferenceEngine& engine);
};


#endif //METATENSOR_MACH7SHAPEINFERENCEBRIDGE_H
