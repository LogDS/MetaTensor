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

#include "logds/libtorchwrapper/CreateMethod.h"

namespace libtorchwrapper {
    CreateMethod::CreateMethod(torch::jit::Module *module, const std::string &final_method_name): closed{false}, final_method_name{final_method_name}, graph{std::make_shared<torch::jit::Graph>()},
        module_(module) {
        // IMPORTANT: The first input to a module method graph MUST be 'self'
        // and it must match the module's internal class type.
        torch::jit::Value* self = graph->addInput("self");
        self->setType(module->_ivalue()->type());
    }

    torch::jit::Value * CreateMethod::declareInput(const std::string &name) {
        auto it = declared_inputs.find(name);
        if (it == declared_inputs.end()) {
            it = declared_inputs.emplace(name, graph->addInput(name)).first;
        }
        return it->second;
    }

    CreateMethod::~CreateMethod() {
        close();
    }

    void CreateMethod::close() {
        if (!closed) {
            // Create a unique qualified name for your module's function
            c10::QualifiedName method_name(*module_->type()->name(), "forward");
            // Compile the graph as a functional method inside the module's compilation unit
            auto method = module_->_ivalue()->compilation_unit()->create_function(method_name, graph);
            // Bind that compiled method directly to the module's type definition
            module_->type()->addMethod(method);
            closed = true;
        }
    }

    void CreateMethod::return_(const std::vector<torch::jit::Value *> &results) const {
        for (const auto& ref : results) {
            graph->registerOutput(ref);
        }
    }

    torch::jit::Value * CreateMethod::binary_comparison(FloatCompOp op, torch::jit::Value *x,
        torch::jit::Value *y) const {
        if ((!x) || (!y)) {
            return nullptr;
        }
        torch::jit::Node * result = nullptr;
        switch (op) {
            case FloatCompOp::LT:
                result = graph->create(torch::jit::aten::lt, {x, y});
                break;
            case FloatCompOp::GT:
                result = graph->create(torch::jit::aten::gt, {x, y});
                break;
            case FloatCompOp::LE:
                result = graph->create(torch::jit::aten::le, {x, y});
                break;
            case FloatCompOp::GE:
                result = graph->create(torch::jit::aten::ge, {x, y});
                break;
            case FloatCompOp::EQ:
                result = graph->create(torch::jit::aten::eq, {x, y});
                break;
            case FloatCompOp::NE:
                result = graph->create(torch::jit::aten::ne, {x, y});
                break;
        }
        if (result) {
            graph->insertNode(result);
            return result->output();
        }
        return nullptr;
    }

    torch::jit::Value * CreateMethod::logical_unary_operator(LogicalUnOp op, torch::jit::Value *x) const {
        if ((!x)) {
            return nullptr;
        }

        torch::jit::Node * result = nullptr;
        switch (op) {
            case LogicalUnOp::NOT:
                result=  graph->create(torch::jit::aten::neg, {x});
                break;
        }
        if (result) {
            graph->insertNode(result);
            return result->output();
        }
        return nullptr;
    }

    torch::jit::Value * CreateMethod::logical_binary_operator(LogicalBinOp op, torch::jit::Value *x,
        torch::jit::Value *y) const {
        if ((!x) || (!y)) {
            return nullptr;
        }

        torch::jit::Node * result = nullptr;
        switch (op) {
            case LogicalBinOp::AND:
                result=   graph->create(torch::jit::aten::logical_and, {x, y});
                break;
            case LogicalBinOp::OR:
                result=   graph->create(torch::jit::aten::logical_or, {x, y});
                break;
            case LogicalBinOp::XOR:
                result=   graph->create(torch::jit::aten::logical_xor, {x, y});
                break;
        }
        if (result) {
            graph->insertNode(result);
            return result->output();
        }
    }
} // libtorchwrapper