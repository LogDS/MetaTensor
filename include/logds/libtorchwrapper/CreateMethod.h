/*
* This file is part of the MetaTensor distribution (https://github.com).
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

#ifndef METATENSOR_CREATEMETHOD_H
#define METATENSOR_CREATEMETHOD_H

#include <torch/script.h>
#include <iostream>

#include "logds/tensormetadata/LambdaAst.h"

namespace libtorchwrapper {
    class CreateMethod {
        std::shared_ptr<torch::jit::Graph> graph;
        std::string final_method_name;
        torch::jit::Module *module_;
        std::unordered_map<std::string, torch::jit::Value*> declared_inputs;
        bool closed;

    public:
        CreateMethod(torch::jit::Module* module, const std::string& final_method_name = "forward");

        CreateMethod(CreateMethod&&) = delete;
        CreateMethod(const CreateMethod&) = default;
        CreateMethod& operator=(CreateMethod&&) = delete;
        CreateMethod& operator=(const CreateMethod&) = default;

            torch::jit::Value* declareInput(const std::string& name);

        ~CreateMethod();

        void close();

        void return_(const std::vector<torch::jit::Value*>& results) const;

        torch::jit::Value* constant(float val) const { return graph->insertConstant(val); }
        torch::jit::Value* constant(int32_t val) const { return graph->insertConstant(val); }
        torch::jit::Value* constant(double val) const { return graph->insertConstant(val); }
        torch::jit::Value* constant(int64_t val) const { return graph->insertConstant(val); }
        torch::jit::Value *binary_comparison(FloatCompOp op, torch::jit::Value* x, torch::jit::Value* y) const;
        torch::jit::Value *logical_unary_operator(LogicalUnOp op, torch::jit::Value* x) const;
        torch::jit::Value *logical_binary_operator(LogicalBinOp op, torch::jit::Value* x, torch::jit::Value* y) const;
    };

} // libtorchwrapper

#endif //METATENSOR_CREATEMETHOD_H
