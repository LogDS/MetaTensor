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


    torch::jit::Value * CreateMethod::constant(const SCALAR &val) const {
        if (std::holds_alternative<torch::jit::Value*>(val)) {
            return std::get<torch::jit::Value*>(val);
        } else if (std::holds_alternative<float>(val)) {
            return constant(std::get<float>(val));
        } else if (std::holds_alternative<double>(val)) {
            return constant(std::get<double>(val));
        } else if (std::holds_alternative<int32_t>(val)) {
            return constant(std::get<int32_t>(val));
        } else if (std::holds_alternative<int64_t>(val)) {
            return constant(std::get<int64_t>(val));
        } else
            return nullptr;
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
        return nullptr;
    }

    torch::jit::Value * CreateMethod::tensor_aggregation(TensorAggregation op, torch::jit::Value *x) const {
        if ((!x)) {
            return nullptr;
        }
        torch::jit::Node * result = nullptr;
        switch (op) {
            case TensorAggregation::ALL:
                result= graph->create(torch::aten::all, {x});
                break;
            case TensorAggregation::ANY:
                result= graph->create(torch::aten::any, {x});
                break;
            case TensorAggregation::ALL_SUM:
                result= graph->create(torch::aten::sum, {x, None(), False(), None()});
                break;
        }
        if (result) {
            graph->insertNode(result);
            return result->output();
        }
        return nullptr;
    }

    torch::jit::Value * CreateMethod::cell_operation(CellOp op, torch::jit::Value *x) {
        if ((!x)) {
            return nullptr;
        }
        torch::jit::Node * result = nullptr;
        switch (op) {
            case CellOp::Sigmoid:
                result = graph->create(torch::aten::sigmoid, {x});
                break;
            case CellOp::Logit:
                result = graph->create(torch::aten::logit, {x});
                break;
            case CellOp::Exp:
                result = graph->create(torch::aten::exp, {x});
                break;
            case CellOp::Exp2:
                result = graph->create(torch::aten::exp2, {x});
                break;
            case CellOp::Log:
                result = graph->create(torch::aten::log, {x});
                break;
            case CellOp::Log2:
                result = graph->create(torch::aten::log2, {x});
                break;
            case CellOp::Log10:
                result = graph->create(torch::aten::log10, {x});
                break;
            case CellOp::Tanh:
                result = graph->create(torch::aten::tanh, {x});
                break;
            case CellOp::Abs:
                result = graph->create(torch::aten::abs, {x});
                break;
            case CellOp::Sqrt:
                result = graph->create(torch::aten::sqrt, {x});
                break;
            case CellOp::Square:
                result = graph->create(torch::aten::square, {x});
                break;
        }
        if (result) {
            graph->insertNode(result);
            return result->output();
        }
        return nullptr;
    }

    torch::jit::Value * CreateMethod::squeeze(torch::jit::Value *x, const std::optional<int64_t> &dim) const {
        torch::jit::Node* node;
        if (dim.has_value()) {
            auto dim_ = constant(dim.value());
            node = graph->create(torch::aten::squeeze, {x, dim_});
        } else {
            node = graph->create(torch::aten::squeeze, {x});
        }
        graph->insertNode(node);
        return node->output();
    }

    torch::jit::Value * CreateMethod::permute(torch::jit::Value *tensor, const std::vector<int64_t> &idx) {
        auto dims_val = graph->insertConstant(c10::List<int64_t>(idx));

        // 2. Crea il nodo aten::permute(Tensor self, int[] dims)
        torch::jit::Node* permute_node = graph->create(torch::jit::aten::permute, {tensor, dims_val});
        graph->insertNode(permute_node);

        return permute_node->output();
    }

    torch::jit::Value * CreateMethod::slice(torch::jit::Value *tensor, const SCALAR&  dim, const SCALAR&  start, const SCALAR&  stop,
        const SCALAR&  step) const {
        auto dim_  = constant(dim);
        auto start_ = constant(start);
        auto stop_ = constant(stop);
        auto step_ = constant(step);
        torch::jit::Node* where_node = graph->create(torch::jit::aten::slice, {tensor, dim_, start_, stop_, step_});
        graph->insertNode(where_node);
        return where_node->output();
    }

    torch::jit::Value * CreateMethod::exists(torch::jit::Value *pred_tensor, std::vector<int64_t> target_dims) const {
        // 1. Ordiniamo in modo decrescente (es: {5, 4, 2})
        std::sort(target_dims.rbegin(), target_dims.rend());
        torch::jit::Value* keepdim_false = False();
        torch::jit::Value* current_tensor = pred_tensor;
        // 2. Applichiamo aten::any direttamente con keepdim=false
        for (int64_t dim : target_dims) {
            torch::jit::Value* dim_val = graph->insertConstant(dim);
            torch::jit::Node* any_node = graph->create(torch::jit::aten::any, {current_tensor, dim_val, keepdim_false});
            graph->insertNode(any_node);
            current_tensor = any_node->output();
        }
        return current_tensor;
    }

    torch::jit::Value * CreateMethod::forall(torch::jit::Value *pred_tensor, std::vector<int64_t> target_dims) const {
        // 1. Ordiniamo in modo decrescente (es: {5, 4, 2})
        std::sort(target_dims.rbegin(), target_dims.rend());
        torch::jit::Value* keepdim_false = False();
        torch::jit::Value* current_tensor = pred_tensor;
        // 2. Applichiamo aten::any direttamente con keepdim=false
        for (int64_t dim : target_dims) {
            torch::jit::Value* dim_val = graph->insertConstant(dim);
            torch::jit::Node* any_node = graph->create(torch::jit::aten::all, {current_tensor, dim_val, keepdim_false});
            graph->insertNode(any_node);
            current_tensor = any_node->output();
        }
        return current_tensor;
    }

    torch::jit::Value * CreateMethod::
    tensor_binary(BinaryTensorOp op, torch::jit::Value *x, torch::jit::Value *y) const {
        if ((!x) || (!y)) {
            return nullptr;
        }
        torch::jit::Node* node = nullptr;
        switch (op) {
            case BinaryTensorOp::MM_2DMatrixMult:
                node = graph->create(torch::aten::mm, {x, y});
                break;
            case BinaryTensorOp::MatMul:
                node = graph->create(torch::aten::matmul, {x, y});
                break;
            case BinaryTensorOp::Cross:
                node = graph->create(torch::aten::cross, {x, y});
                break;
            case BinaryTensorOp::ADD:
                node = graph->create(torch::aten::add, {x, y});
                break;
            case BinaryTensorOp::ElementWise_MUL:
                node = graph->create(torch::aten::mul, {x, y});
                break;
        }
        if (node) {
            graph->insertNode(node);
            return node->output();
        }
        return nullptr;
    }
} // libtorchwrapper