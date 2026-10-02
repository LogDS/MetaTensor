//
// Created by gyankos on 02/10/26.
//

#ifndef METATENSOR_MACH7SHAPEINFERENCEBRIDGE_H
#define METATENSOR_MACH7SHAPEINFERENCEBRIDGE_H


#include <logds/tensormetadata/Metadata.h>
#include <logds/tensormetadata/JitShapeInferenceEngine.h>
#include <optional>
#include <iostream>


class Mach7ShapeInferenceBridge {
public:
    // RISOLUTIVO: Il metodo accetta l'istanza dell'engine per accumulare i vincoli di unificazione contestuale!
    static std::optional<JitTensorMetadata> infer_node_shape(AstNode* node, const JitShapeInferenceEngine& engine);
};


#endif //METATENSOR_MACH7SHAPEINFERENCEBRIDGE_H
