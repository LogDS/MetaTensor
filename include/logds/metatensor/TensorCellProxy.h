//
// Created by gyankos on 01/10/26.
//

#ifndef METATENSOR_TENSORCELLPROXY_H
#define METATENSOR_TENSORCELLPROXY_H

#include <torch/torch.h>
#include <cstdint>

// =============================================================================
// PROXY INTERMEDIO PER L'ACCESSO E MUTAZIONE DELLA MEMORIA (C++26)
// =============================================================================
template <typename Scalar>
struct TensorCellProxy {
    torch::Tensor& parent_storage;
    int64_t linear_index;

    // Assegnazione in-place sulla GPU/CPU (tensor[coords] = value;)
    TensorCellProxy& operator=(Scalar value) {
        torch::NoGradGuard no_grad;
        auto scalar_tensor = torch::tensor(value, parent_storage.options());
        parent_storage.flatten()[linear_index].copy_(scalar_tensor);
        return *this;
    }

    // Cast implicito per l'estrazione (float val = tensor[coords];)
    operator Scalar() const {
        return parent_storage.flatten()[linear_index].item<float>();
    }
};


#endif //METATENSOR_TENSORCELLPROXY_H
