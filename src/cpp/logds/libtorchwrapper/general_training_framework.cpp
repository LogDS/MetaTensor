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

#include "logds/libtorchwrapper/general_training_framework.h"

#include <torch/script.h>
#include <vector>
#include <torch/script.h>
#include <torch/csrc/jit/ir/ir.h>            // Fornisce l'accesso completo a torch::jit::Graph, Block e Node
#include <torch/csrc/jit/passes/inliner.h>   // Se necessario per le utility di inlineamento


#include <torch/script.h>
#include <vector>

#include "logds/metatensor/GradientTape.h"


#include <torch/script.h>
#include <torch/csrc/jit/ir/ir.h>
#include <vector>


torch::jit::Value* createTupleResult(torch::jit::Graph* graph,
                                     const std::vector<torch::jit::Value*>& inputs,
                                     const std::vector<c10::TypePtr>& types){
    if (!graph)
        return nullptr;
    // 1. Create a prim::TupleConstruct node inside the graph
    torch::jit::Node* tuple_node = graph->create(torch::jit::prim::TupleConstruct,
                                                 inputs);
    // 2. Append the node to the block so it is executed
    graph->appendNode(tuple_node);

    // 3. Register the single Tuple output as the graph's overall return value
    // Assegnazione del tipo di tupla pura non nominale
    auto pure_tuple_type = c10::TupleType::create(types);
    auto tuple = tuple_node->output();
    tuple->setType(pure_tuple_type);
    graph->registerOutput(tuple);
    return tuple;
}

std::vector<torch::jit::Value*> extractTupleResult(torch::jit::Value* combined_tuple_out,
                                                   torch::jit::Graph* master_graph,
                                                   uint64_t expected_size,
    const std::vector<c10::TypePtr>& expected_types) {
    // Assume 'master_graph' is your outer block, and 'call_node' is a prim::CallGraph node
    std::vector<torch::jit::Value*> result;
    // Create a structural unpack node in the master graph
    torch::jit::Node* unpack_node = master_graph->create(torch::jit::prim::TupleUnpack, {combined_tuple_out});

    // Configure the correct size slots for downstream compilation mapping
    for (size_t i = 0; i < expected_size; ++i) {
        unpack_node->addOutput()->setType(expected_types[i]);;
    }
    master_graph->appendNode(unpack_node);

    // Retrieve the separated outputs to wire into the rest of your pipeline
    for (uint64_t idx = 0; idx < expected_size; ++idx) {
        result.emplace_back(unpack_node->output(idx));
    }
    return result;
}

std::shared_ptr<torch::jit::Graph> create_parametrizable_optimizer_subgraph(
    size_t num_tensors,
    OptimizerType opt_type,
    DecayType decay_type,
    float gamma_val_init,
    int64_t decay_steps_init
) {
    auto g = std::make_shared<torch::jit::Graph>();

    // =========================================================================
    // 1. CANALI DI INGRESSO (INPUTS SIMBOLICI TIPIZZATI)
    // =========================================================================
    torch::jit::Value* weights_in = g->addInput("weights_in");   // Tensor[]
    torch::jit::Value* grads_in   = g->addInput("grads_in");     // Tensor[]
    torch::jit::Value* lr_in      = g->addInput("lr_in");        // float (Scalar)
    torch::jit::Value* m_in       = g->addInput("m_in");         // Tensor[]
    torch::jit::Value* v_in       = g->addInput("v_in");         // Tensor[]
    torch::jit::Value* loop_iter  = g->addInput("loop_iter");    // int (Scalar)

    auto tensor_type = c10::TensorType::get();
    auto tensor_list_type = c10::ListType::ofTensors();
    weights_in->setType(tensor_list_type);
    grads_in->setType(tensor_list_type);
    m_in->setType(tensor_list_type);
    v_in->setType(tensor_list_type);
    lr_in->setType(c10::FloatType::get());
    loop_iter->setType(c10::IntType::get());

    torch::jit::Value* alpha_one = g->insertConstant(1.0);


    std::vector<torch::jit::Value*> updated_weights;
    std::vector<torch::jit::Value*> updated_m;
    std::vector<torch::jit::Value*> updated_v;

    // =========================================================================
    // 2. OTTIMIZZATORE: STRUTTURAZIONE OPERAZIONI MATEMATICHE (Senza nodi duplicati)
    // =========================================================================
    for (size_t i = 0; i < num_tensors; ++i) {
        torch::jit::Value* idx = g->insertConstant(static_cast<int64_t>(i));

        torch::jit::Node* w_i = g->create(torch::jit::aten::select, {weights_in, idx});
        torch::jit::Node* g_i = g->create(torch::jit::aten::select, {grads_in, idx});
        torch::jit::Node* m_i = g->create(torch::jit::aten::select, {m_in, idx});
        torch::jit::Node* v_i = g->create(torch::jit::aten::select, {v_in, idx});

        g->insertNode(w_i); g->insertNode(g_i); g->insertNode(m_i); g->insertNode(v_i);

        w_i->output()->setType(tensor_type);
        g_i->output()->setType(tensor_type);
        m_i->output()->setType(tensor_type);
        v_i->output()->setType(tensor_type);

        torch::jit::Value* next_w = nullptr;
        torch::jit::Value* next_m = m_i->output();
        torch::jit::Value* next_v = v_i->output();

        if (opt_type == OptimizerType::SGD) {
            torch::jit::Node* step = g->create(torch::jit::aten::mul, {g_i->output(), lr_in});
            g->insertNode(step);
            torch::jit::Node* upd = g->create(torch::jit::aten::sub, {w_i->output(), step->output(), alpha_one});
            g->insertNode(upd);
            next_w = upd->output();
        }
        else if (opt_type == OptimizerType::Momentum) {
            torch::jit::Value* momentum_factor = g->insertConstant(0.9);
            torch::jit::Node* v_sc = g->create(torch::jit::aten::mul, {m_i->output(), momentum_factor});
            g->insertNode(v_sc);
            torch::jit::Node* v_nx = g->create(torch::jit::aten::add, {v_sc->output(), g_i->output(), alpha_one});
            g->insertNode(v_nx);
            next_m = v_nx->output();

            torch::jit::Node* step = g->create(torch::jit::aten::mul, {next_m, lr_in});
            g->insertNode(step);
            torch::jit::Node* upd = g->create(torch::jit::aten::sub, {w_i->output(), step->output(), alpha_one});
            g->insertNode(upd);
            next_w = upd->output();
        }
        else if (opt_type == OptimizerType::Adam) {
            torch::jit::Value* b1 = g->insertConstant(0.9);
            torch::jit::Value* b2 = g->insertConstant(0.999);
            torch::jit::Value* eps = g->insertConstant(1e-8);

            torch::jit::Node* m_nx = g->create(torch::jit::aten::lerp, {g_i->output(), m_i->output(), b1});
            next_m = m_nx->output();
            next_m->setType(tensor_type);
            g->insertNode(m_nx);

            torch::jit::Node* grad_sq = g->create(torch::jit::aten::mul, {g_i->output(), g_i->output()});
            g->insertNode(grad_sq);
            torch::jit::Node* v_nx = g->create(torch::jit::aten::lerp, {grad_sq->output(), v_i->output(), b2});
            g->insertNode(v_nx);
            next_v = v_nx->output();
            next_v->setType(tensor_type);

            torch::jit::Node* sq_v = g->create(torch::jit::aten::sqrt, {next_v});
            g->insertNode(sq_v);
            torch::jit::Node* den = g->create(torch::jit::aten::add, {sq_v->output(), eps, alpha_one});
            g->insertNode(den);
            torch::jit::Node* stp = g->create(torch::jit::aten::div, {next_m, den->output()});
            stp->output()->setType(c10::TensorType::get());
            g->insertNode(stp);
            torch::jit::Node* lr_stp = g->create(torch::jit::aten::mul, {stp->output(), lr_in});
            g->insertNode(lr_stp);
            torch::jit::Node* upd = g->create(torch::jit::aten::sub, {w_i->output(), lr_stp->output(), alpha_one});
            upd->output()->setType(c10::TensorType::get());
            next_w = upd->output();
            next_w->setType(tensor_type);
            g->insertNode(upd);
        }

        updated_weights.push_back(next_w);
        updated_m.push_back(next_m);
        updated_v.push_back(next_v);
    }

    torch::jit::Node* next_weights_list = g->create(torch::jit::prim::ListConstruct, updated_weights);
    torch::jit::Node* next_m_list       = g->create(torch::jit::prim::ListConstruct, updated_m);
    torch::jit::Node* next_v_list       = g->create(torch::jit::prim::ListConstruct, updated_v);
    next_weights_list->output()->setType(tensor_list_type);
    next_m_list->output()->setType(tensor_list_type);
    next_v_list->output()->setType(tensor_list_type);
    g->insertNode(next_weights_list);
    g->insertNode(next_m_list);
    g->insertNode(next_v_list);

    // =========================================================================
    // 3. LOGICA DI DECAY DEL LEARNING RATE
    // =========================================================================
    torch::jit::Value* next_lr = lr_in;
    torch::jit::Value* gamma_val = g->insertConstant(gamma_val_init);

    if (decay_type == DecayType::Exponential) {
        torch::jit::Node* lr_decay = g->create(torch::jit::aten::mul, {lr_in, gamma_val});
        g->insertNode(lr_decay);
        next_lr = lr_decay->output();
    }
    else if (decay_type == DecayType::Step) {
        torch::jit::Value* steps_val = g->insertConstant(decay_steps_init);
        torch::jit::Node* mod_node = g->create(torch::jit::aten::remainder, {loop_iter, steps_val});
        g->insertNode(mod_node);
        torch::jit::Node* zero_check = g->create(torch::jit::aten::eq, {mod_node->output(), g->insertConstant(static_cast<int64_t>(0))});
        g->insertNode(zero_check);

        torch::jit::Node* if_node = g->create(torch::jit::prim::If, {zero_check->output()});
        g->insertNode(if_node);

        // Ramo Then
        torch::jit::Block* then_block = if_node->addBlock();
        {
            torch::jit::WithInsertPoint then_ip(then_block);
            torch::jit::Node* lr_decay = g->create(torch::jit::aten::mul, {lr_in, gamma_val});
            // CORREZIONE: g->insertNode rimosso. Il nodo viene inserito in automatico nel blocco corretto!
            then_block->registerOutput(lr_decay->output());
        }

        // Ramo Else
        torch::jit::Block* else_block = if_node->addBlock();
        {
            torch::jit::WithInsertPoint else_ip(else_block);
            else_block->registerOutput(lr_in);
        }

        next_lr = if_node->output();
        next_lr->setType(c10::FloatType::get());
    }

    // =========================================================================
    // 4. CANALI DI USCITA
    // =========================================================================
    createTupleResult(g.get(), {next_weights_list->output(), next_lr, next_m_list->output(), next_v_list->output()}, {} ); // TODO: pass the types at the very right
    return g;
}


#include "logds/libtorchwrapper/general_training_framework.h"
#include <torch/script.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/passes/inliner.h>
#include <vector>
#include <memory>

// (Mantieni qui le tue definizioni corrette di createTupleResult ed extractTupleResult)
#include "logds/libtorchwrapper/general_training_framework.h"
#include <torch/script.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/passes/inliner.h>
#include <vector>
#include <memory>
#include "logds/libtorchwrapper/general_training_framework.h"
#include <torch/script.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/passes/inliner.h>
#include <vector>
#include <memory>
std::shared_ptr<torch::jit::Graph> build_universal_training_pipeline(
    std::shared_ptr<torch::jit::Graph> user_step_subgraph,
    std::shared_ptr<torch::jit::Graph> optimizer_subgraph,
    size_t num_tensors,
    int max_epochs,
    float initial_lr
) {
    auto main_graph = std::make_shared<torch::jit::Graph>();

    // 1. Configurazione Canali di Ingresso Radice
    torch::jit::Value* in_dataset_inputs  = main_graph->addInput("dataset_inputs");
    torch::jit::Value* in_dataset_targets = main_graph->addInput("dataset_targets");
    torch::jit::Value* in_initial_weights = main_graph->addInput("initial_weights");

    auto tensor_list_type = c10::ListType::ofTensors();
    in_dataset_inputs->setType(tensor_list_type);
    in_dataset_targets->setType(tensor_list_type);
    in_initial_weights->setType(tensor_list_type);

    // Costanti e configurazioni strutturali
    torch::jit::Value* loop_limit_huge = main_graph->insertConstant(static_cast<int64_t>(9223372036854775807));
    torch::jit::Value* cond_true       = main_graph->insertConstant(true);
    torch::jit::Value* max_epochs_val  = main_graph->insertConstant(static_cast<int64_t>(max_epochs));
    torch::jit::Value* epoch_start_val = main_graph->insertConstant(static_cast<int64_t>(0));
    torch::jit::Value* lr_init_val     = main_graph->insertConstant(initial_lr);
    torch::jit::Value* none_val        = main_graph->insertConstant(c10::nullopt);
    torch::jit::Value* int_one         = main_graph->insertConstant(static_cast<int64_t>(1));

    // Allocazione stati ottimizzatore (m_list, v_list) tramite unrolling
    std::vector<torch::jit::Value*> zero_m_elements;
    std::vector<torch::jit::Value*> zero_v_elements;
    for (size_t i = 0; i < num_tensors; ++i) {
        torch::jit::Value* idx = main_graph->insertConstant(static_cast<int64_t>(i));
        torch::jit::Node* w_i = main_graph->create(torch::jit::aten::select, {in_initial_weights, idx});
        main_graph->insertNode(w_i);
        w_i->output()->setType(c10::TensorType::get());

        torch::jit::Node* z_m = main_graph->create(torch::jit::aten::zeros_like, {w_i->output(), none_val, none_val, none_val, none_val, none_val});
        torch::jit::Node* z_v = main_graph->create(torch::jit::aten::zeros_like, {w_i->output(), none_val, none_val, none_val, none_val, none_val});
        main_graph->insertNode(z_m); main_graph->insertNode(z_v);

        zero_m_elements.push_back(z_m->output());
        zero_v_elements.push_back(z_v->output());
    }
    torch::jit::Value* init_m = main_graph->create(torch::jit::prim::ListConstruct, zero_m_elements)->output();
    torch::jit::Value* init_v = main_graph->create(torch::jit::prim::ListConstruct, zero_v_elements)->output();
    main_graph->insertNode(main_graph->create(torch::jit::prim::ListConstruct, zero_m_elements));
    main_graph->insertNode(main_graph->create(torch::jit::prim::ListConstruct, zero_v_elements));
    init_m->setType(tensor_list_type); init_v->setType(tensor_list_type);

    // =========================================================================
    // ACCORPAMENTO IN TUPLE DEGLI STATI (Soddisfa la regola dei 5 parametri)
    // =========================================================================
    // Stato 2 (Ottimizzatore): Conterrà [m_list, v_list]
    torch::jit::Node* opt_state_tuple = main_graph->create(torch::jit::prim::TupleConstruct, {init_m, init_v});
    main_graph->insertNode(opt_state_tuple);

    // Stato 3 (Iperparametri): Conterrà [lr, epoch_iter]
    torch::jit::Node* hyper_state_tuple = main_graph->create(torch::jit::prim::TupleConstruct, {lr_init_val, epoch_start_val});
    main_graph->insertNode(hyper_state_tuple);

    // 2. Istanziazione prim::Loop rigorosamente con 5 parametri
    std::vector<torch::jit::Value*> loop_inputs = {
        loop_limit_huge, cond_true, in_initial_weights, opt_state_tuple->output(), hyper_state_tuple->output()
    };
    torch::jit::Node* loop_node = main_graph->create(torch::jit::prim::Loop, loop_inputs);
    main_graph->insertNode(loop_node);

    // Configurazione del Blocco (Accetta rigorosamente 4 ingressi strutturali)
    torch::jit::Block* body = loop_node->addBlock();
    torch::jit::Value* loop_index_jit = body->addInput("loop_index_jit"); // Slot 0
    torch::jit::Value* loop_cond_in   = body->addInput("cond_in");        // Slot 1
    torch::jit::Value* b_weights      = body->addInput("weights_list");   // Slot 2 (Tensor[])
    torch::jit::Value* b_opt_tuple    = body->addInput("opt_tuple");      // Slot 3 (Tuple)
    torch::jit::Value* b_hyper_tuple  = body->addInput("hyper_tuple");    // Slot 4 (Tuple)

    loop_index_jit->setType(c10::IntType::get());
    loop_cond_in->setType(c10::BoolType::get());
    b_weights->setType(tensor_list_type);

    torch::jit::WithInsertPoint insert_point(body);

    // Spacchettamento delle Tuple all'ingresso del ciclo per alimentare i sottografi
    std::vector<c10::TypePtr> opt_types = { tensor_list_type, tensor_list_type };
    auto opt_unpacked = extractTupleResult(b_opt_tuple, main_graph.get(), 2, opt_types);
    torch::jit::Value* b_m_list = opt_unpacked[0];
    torch::jit::Value* b_v_list = opt_unpacked[1];

    std::vector<c10::TypePtr> hyper_types = { c10::FloatType::get(), c10::IntType::get() };
    auto hyper_unpacked = extractTupleResult(b_hyper_tuple, main_graph.get(), 2, hyper_types);
    torch::jit::Value* b_lr         = hyper_unpacked[0];
    torch::jit::Value* b_epoch_iter = hyper_unpacked[1];

    // =========================================================================
    // INNESTO DEI SOTTOGRAFI (Il core invariato del tuo framework)
    // =========================================================================
    std::vector<torch::jit::Value*> user_inputs = {
        b_weights, in_dataset_inputs, in_dataset_targets, b_epoch_iter, loop_cond_in
    };
    std::vector<torch::jit::Value*> uo = torch::jit::insertGraph(*main_graph, *user_step_subgraph, user_inputs);
    std::vector<c10::TypePtr> user_expected_types = { c10::TensorType::get(), c10::BoolType::get() };
    auto user_outputs = extractTupleResult(uo[0], main_graph.get(), 2, user_expected_types);
    torch::jit::Value* current_loss       = user_outputs[0];
    torch::jit::Value* user_stopping_cond = user_outputs[1];

    // Autograd
    torch::jit::Node* outputs_list = main_graph->create(torch::jit::prim::ListConstruct, {current_loss});
    body->appendNode(outputs_list);
    torch::jit::Node* grad_node = main_graph->create(torch::jit::aten::grad, {
        outputs_list->output(), b_weights, none_val, none_val, none_val, main_graph->insertConstant(false)
    });
    body->appendNode(grad_node);
    torch::jit::Value* computed_gradients = grad_node->output();
    computed_gradients->setType(tensor_list_type);

    // Ottimizzatore
    std::vector<torch::jit::Value*> optimizer_inputs = {
        b_weights, computed_gradients, b_lr, b_m_list, b_v_list, b_epoch_iter
    };
    std::vector<torch::jit::Value*> oo = torch::jit::insertGraph(*main_graph, *optimizer_subgraph, optimizer_inputs);
    std::vector<c10::TypePtr> optimizer_expected_types = { tensor_list_type, c10::FloatType::get(), tensor_list_type, tensor_list_type };
    auto optimizer_outputs = extractTupleResult(oo[0], main_graph.get(), 4, optimizer_expected_types);
    torch::jit::Value* next_weights_list = optimizer_outputs[0];
    torch::jit::Value* next_lr           = optimizer_outputs[1];
    torch::jit::Value* next_m_list       = optimizer_outputs[2];
    torch::jit::Value* next_v_list       = optimizer_outputs[3];

    // =========================================================================
    // CALCOLO AVANZAMENTO EPOCHE ED AND LOGICO NATIVO
    // =========================================================================
    torch::jit::Node* next_epoch_node = main_graph->create(torch::jit::aten::add, {b_epoch_iter, int_one, int_one});
    body->appendNode(next_epoch_node);
    torch::jit::Value* next_epoch_val = next_epoch_node->output();
    next_epoch_val->setType(c10::IntType::get());

    torch::jit::Node* epoch_check_node = main_graph->create(torch::jit::aten::lt, {next_epoch_val, max_epochs_val});
    body->appendNode(epoch_check_node);

    torch::jit::Node* and_node = main_graph->create(torch::jit::aten::__and__, {epoch_check_node->output(), user_stopping_cond});
    body->appendNode(and_node);
    torch::jit::Value* next_loop_cond = and_node->output();
    next_loop_cond->setType(c10::BoolType::get());

    // =========================================================================
    // RI-IMPACCHETTAMENTO DELLE TUPLE DI USCITA (Esattamente 4 canali registrati!)
    // =========================================================================
    torch::jit::Node* out_opt_tuple = main_graph->create(torch::jit::prim::TupleConstruct, {next_m_list, next_v_list});
    body->appendNode(out_opt_tuple);

    torch::jit::Node* out_hyper_tuple = main_graph->create(torch::jit::prim::TupleConstruct, {next_lr, next_epoch_val});
    body->appendNode(out_hyper_tuple);

    // Registrazione speculare millimetrica con block0
    body->registerOutput(next_loop_cond);       // Mappa su loop_cond_in
    body->registerOutput(next_weights_list);   // Mappa su b_weights
    body->registerOutput(out_opt_tuple->output());   // Mappa su b_opt_tuple
    body->registerOutput(out_hyper_tuple->output()); // Mappa su b_hyper_tuple

    // Estrazione finale all'esterno del ciclo
    torch::jit::Value* final_weights_out = loop_node->output(0);
    final_weights_out->setType(tensor_list_type);
    main_graph->registerOutput(final_weights_out);

    return main_graph;
}
