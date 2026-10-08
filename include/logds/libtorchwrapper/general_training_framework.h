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
 * along with this program. If not, see <http://gnu.org>.
 */

#ifndef METATENSOR_GENERAL_TRAINING_FRAMEWORK_H
#define METATENSOR_GENERAL_TRAINING_FRAMEWORK_H

#include <memory>
#include "logds/metatensor/GradientTape.h"

torch::jit::Value* createTupleResult(torch::jit::Graph* graph,
                                     const std::vector<torch::jit::Value*>& inputs,
                                     const std::vector<c10::TypePtr>& types);

std::shared_ptr<torch::jit::Graph> create_parametrizable_optimizer_subgraph(
    size_t num_tensors,
    OptimizerType opt_type,
    DecayType decay_type,
    float gamma_val_init = 0.1f,
    int64_t decay_steps_init = 10
);

void build_universal_training_pipeline(
    std::shared_ptr<torch::jit::Graph>& main_graph,
    const std::vector<torch::jit::Value*>& dataset_inputs_vector,
    const std::vector<torch::jit::Value*>& dataset_targets_vector,
    const std::vector<torch::jit::Value*>& initial_weights,
    std::shared_ptr<torch::jit::Graph> user_step_subgraph,
    std::shared_ptr<torch::jit::Graph> optimizer_subgraph,
    int max_epochs,
    float initial_lr
);

#endif //METATENSOR_GENERAL_TRAINING_FRAMEWORK_H
