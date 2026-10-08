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

#ifndef METATENSOR_CREATEMETHOD_H
#define METATENSOR_CREATEMETHOD_H

#include <torch/script.h>
#include <iostream>

#include "logds/metatensor/CellOp.h"
#include "logds/tensormetadata/LambdaAst.h"

using SCALAR = std::variant<float, double, int64_t, int32_t, torch::jit::Value*>;

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
        auto None() const  { return graph->insertConstant(c10::nullopt);  }
        auto True() const  { return graph->insertConstant(true);          }
        auto False() const { return graph->insertConstant(false);         }

        torch::jit::Value* constant(float val) const        { return graph->insertConstant(val); }
        torch::jit::Value* constant(int32_t val) const      { return graph->insertConstant(val); }
        torch::jit::Value* constant(double val) const       { return graph->insertConstant(val); }
        torch::jit::Value* constant(int64_t val) const      { return graph->insertConstant(val); }
        torch::jit::Value* constant(const SCALAR& val) const;

        torch::jit::Value* binary_comparison(FloatCompOp op, torch::jit::Value* x, torch::jit::Value* y) const;
        torch::jit::Value* logical_unary_operator(LogicalUnOp op, torch::jit::Value* x) const;
        torch::jit::Value* logical_binary_operator(LogicalBinOp op, torch::jit::Value* x, torch::jit::Value* y) const;
        torch::jit::Value* tensor_aggregation(TensorAggregation op, torch::jit::Value* x) const;
        torch::jit::Value* cell_operation(CellOp op, torch::jit::Value* x);
        torch::jit::Value* squeeze(torch::jit::Value* x, const std::optional<int64_t>& dim) const;
        torch::jit::Value* permute(torch::jit::Value* tensor, const std::vector<int64_t>& idx);
        torch::jit::Value* slice(torch::jit::Value* tensor, const SCALAR&  dim, const SCALAR&  start, const SCALAR&  stop, const SCALAR&  step = 1) const;
        torch::jit::Value* exists(torch::jit::Value* pred_tensor, std::vector<int64_t> target_dims) const;
        torch::jit::Value* forall(torch::jit::Value* pred_tensor, std::vector<int64_t> target_dims) const;
        torch::jit::Value* tensor_binary(BinaryTensorOp op, torch::jit::Value* x, torch::jit::Value* y) const;

        template<typename T>
        torch::jit::Value* clip(torch::jit::Value* tensor, T min, T max) const {
            auto min_ = constant(min);
            auto max_ = constant(max);
            torch::jit::Node* full_true_node = graph->create(torch::jit::aten::clamp,{tensor, min_, max_});
            graph->insertNode(full_true_node);
            return full_true_node->output();
        }

        template<typename T>
        torch::jit::Value *where(torch::jit::Value* true_false_tensor, T constant_true, T constant_false) const {
            // CREAZIONE DEI VALORI COSTANTI CON full_like
            // Schema: aten::full_like(Tensor self, Scalar fill_value, ...)
            // =========================================================================
            torch::jit::Value* val_true = constant(constant_true);  // Valore se True
            torch::jit::Value* val_false = constant(constant_false); // Valore se False

            // I restanti parametri di full_like possono essere lasciati a None (c10::nullopt)
            // in modo da ereditare automaticamente dtype, layout e device da X.
            torch::jit::Value* none_val = None();

            // Tensore per il ramo True (riempito con 1.0)
            torch::jit::Node* full_true_node = graph->create(torch::jit::aten::full_like,
                {true_false_tensor, val_true, none_val, none_val, none_val, none_val, none_val});
            graph->insertNode(full_true_node);
            torch::jit::Value* x_true = full_true_node->output();

            // Tensore per il ramo False (riempito con -1.0)
            torch::jit::Node* full_false_node = graph->create(torch::jit::aten::full_like,
                {true_false_tensor, val_false, none_val, none_val, none_val, none_val, none_val});
            graph->insertNode(full_false_node);
            torch::jit::Value* x_false = full_false_node->output();

            // =========================================================================
            // CREAZIONE DEL NODO aten::where
            // Schema: aten::where(Tensor condition, Tensor self, Tensor other) -> Tensor
            // =========================================================================
            torch::jit::Node* where_node = graph->create(torch::jit::aten::where, {true_false_tensor, x_true, x_false});
            graph->insertNode(where_node);
            return where_node->output();
        }
    };
} // libtorchwrapper

#endif //METATENSOR_CREATEMETHOD_H
