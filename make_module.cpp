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

#include <torch/script.h>
#include <iostream>

#include "logds/libtorchwrapper/CreateLibTorchModule.h"
#include "logds/tensormetadata/LambdaAst.h"

int main() {
    // 1. Instantiate an empty JIT module
    libtorchwrapper::CreateLibTorchModule module("FilterModule");
    {
        auto forward = module.createMethod("forward");

        // 2. Define your computational graph
        // auto graph = std::make_shared<torch::jit::Graph>();
        // 3. Add your actual tensor input 'X'
        auto x = forward.declareInput("x");
        // torch::jit::Value* x = graph->addInput("x");

        // 4. Construct your expression logic: (X > 0.5 || X < -35.7)
        auto thresh_upper = forward.constant(0.5);
        auto thresh_lower = forward.constant(-35.7);

        auto gt = forward.binary_comparison(FloatCompOp::GT, x, thresh_upper);
        auto lt = forward.binary_comparison(FloatCompOp::LT, x, thresh_lower);
        auto or_node = forward.logical_binary_operator(LogicalBinOp::OR, gt, lt);
        forward.return_({or_node});
        forward.close();
    }

    // Finalize the graph's output
    // =========================================================================
    // 5. Test executing the newly attached method
    // torch::Tensor input_x = torch::tensor({1.0, 0.1, -40.0, -10.0});
    // Pass the standard vector stack to execute the method
    // std::vector<torch::jit::IValue> inputs = {input_x};
    // auto output = module.forward(inputs).toTensor();
    // std::cout << "Successfully executed C++ JIT Graph method:\n" << output << std::endl;
    // ... (All previous graph generation and method registration code) ...

    // 6. Dump the generated module to disk
    std::string model_path = "generated_cpp_module.pt";
    module.compile(model_path);


    std::cout << "Module successfully dumped to " << model_path << std::endl;

}
