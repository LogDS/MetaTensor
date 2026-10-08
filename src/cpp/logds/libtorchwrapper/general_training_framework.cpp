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

    auto tensor_list_type = c10::ListType::create(c10::TensorType::get());
    weights_in->setType(tensor_list_type);
    grads_in->setType(tensor_list_type);
    m_in->setType(tensor_list_type);
    v_in->setType(tensor_list_type);
    lr_in->setType(c10::FloatType::get());
    loop_iter->setType(c10::IntType::get());

    torch::jit::Value* alpha_one = g->insertConstant(1.0);
    auto tensor_type = c10::TensorType::get();

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
            g->insertNode(m_nx);
            next_m = m_nx->output();

            torch::jit::Node* grad_sq = g->create(torch::jit::aten::mul, {g_i->output(), g_i->output()});
            g->insertNode(grad_sq);
            torch::jit::Node* v_nx = g->create(torch::jit::aten::lerp, {grad_sq->output(), v_i->output(), b2});
            g->insertNode(v_nx);
            next_v = v_nx->output();

            torch::jit::Node* sq_v = g->create(torch::jit::aten::sqrt, {next_v});
            g->insertNode(sq_v);
            torch::jit::Node* den = g->create(torch::jit::aten::add, {sq_v->output(), eps, alpha_one});
            g->insertNode(den);
            torch::jit::Node* stp = g->create(torch::jit::aten::div, {next_m, den->output()});
            g->insertNode(stp);
            torch::jit::Node* lr_stp = g->create(torch::jit::aten::mul, {stp->output(), lr_in});
            g->insertNode(lr_stp);
            torch::jit::Node* upd = g->create(torch::jit::aten::sub, {w_i->output(), lr_stp->output(), alpha_one});
            g->insertNode(upd);
            next_w = upd->output();
        }

        updated_weights.push_back(next_w);
        updated_m.push_back(next_m);
        updated_v.push_back(next_v);
    }

    torch::jit::Node* next_weights_list = g->create(torch::jit::prim::ListConstruct, updated_weights);
    torch::jit::Node* next_m_list       = g->create(torch::jit::prim::ListConstruct, updated_m);
    torch::jit::Node* next_v_list       = g->create(torch::jit::prim::ListConstruct, updated_v);
    g->insertNode(next_weights_list); g->insertNode(next_m_list); g->insertNode(next_v_list);

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
    }

    // =========================================================================
    // 4. CANALI DI USCITA
    // =========================================================================
    createTupleResult(g.get(), {next_weights_list->output(), next_lr, next_m_list->output(), next_v_list->output()}, {} ); // TODO: pass the types at the very right
    return g;
}

void build_universal_training_pipeline(
    std::shared_ptr<torch::jit::Graph>& main_graph,
    const std::vector<torch::jit::Value*>& dataset_inputs_vector,
    const std::vector<torch::jit::Value*>& dataset_targets_vector,
    const std::vector<torch::jit::Value*>& initial_weights,
    std::shared_ptr<torch::jit::Graph> user_step_subgraph,
    std::shared_ptr<torch::jit::Graph> optimizer_subgraph,
    int max_epochs,
    float initial_lr
) {
    // 1. Inizializzazione delle costanti esterne al ciclo
    torch::jit::Value* max_epochs_val = main_graph->insertConstant(max_epochs);
    torch::jit::Value* cond_true = main_graph->insertConstant(true);
    torch::jit::Value* lr_init_val = main_graph->insertConstant(initial_lr);
    torch::jit::Value* none_val = main_graph->insertConstant(c10::nullopt);

    // Impacchettamento dei vettori di dataset in Liste JIT native (Tensor[])
    torch::jit::Node* inputs_list_node = main_graph->create(torch::jit::prim::ListConstruct, dataset_inputs_vector);
    torch::jit::Node* targets_list_node = main_graph->create(torch::jit::prim::ListConstruct, dataset_targets_vector);
    main_graph->insertNode(inputs_list_node);
    main_graph->insertNode(targets_list_node);

    torch::jit::Value* dataset_inputs_list = inputs_list_node->output();
    torch::jit::Value* dataset_targets_list = targets_list_node->output();

    // Impacchettamento iniziale dei parametri globali da ottimizzare
    torch::jit::Node* w_list_node = main_graph->create(torch::jit::prim::ListConstruct, initial_weights);
    main_graph->insertNode(w_list_node);
    torch::jit::Value* initial_weights_list = w_list_node->output();

    // Allocazione degli stati dell'ottimizzatore (M1, M2) come liste di zeri
    std::vector<torch::jit::Value*> zero_tensors;
    for (auto* w : initial_weights) {
        torch::jit::Node* z = main_graph->create(torch::jit::aten::zeros_like, {w, none_val, none_val, none_val, none_val, none_val});
        main_graph->insertNode(z);
        zero_tensors.push_back(z->output());
    }

    torch::jit::Node* m_list_node = main_graph->create(torch::jit::prim::ListConstruct, zero_tensors);
    torch::jit::Node* v_list_node = main_graph->create(torch::jit::prim::ListConstruct, zero_tensors);
    main_graph->insertNode(m_list_node);
    main_graph->insertNode(v_list_node);

    torch::jit::Value* init_m = m_list_node->output();
    torch::jit::Value* init_v = v_list_node->output();

    // 2. Registrazione delle Loop-Carried Variables in ingresso a prim::Loop
    std::vector<torch::jit::Value*> loop_inputs = {
        max_epochs_val, cond_true, initial_weights_list, lr_init_val, init_m, init_v
    };
    torch::jit::Node* loop_node = main_graph->create(torch::jit::prim::Loop, loop_inputs);
    main_graph->insertNode(loop_node);

    // 3. Configurazione del Blocco Interno del Ciclo (Corpo del Loop)
    torch::jit::Block* body = loop_node->addBlock();
    torch::jit::Value* loop_iter    = body->addInput("iter");
    torch::jit::Value* loop_cond_in = body->addInput("cond_in");
    torch::jit::Value* b_weights    = body->addInput("weights_list");
    torch::jit::Value* b_lr         = body->addInput("lr");
    torch::jit::Value* b_m_list     = body->addInput("m_list");
    torch::jit::Value* b_v_list     = body->addInput("v_list");

    // CORREZIONE FONDAMENTALE: Impostiamo l'insert point direttamente all'interno del blocco body.
    // Usiamo il blocco per forzare il corretto accoppiamento sequenziale.
    torch::jit::WithInsertPoint insert_point(body);

    // =========================================================================
    // PASSO 1: INNESTO DEL SOTTOGRAFO DELL'UTENTE (All'interno di body)
    // =========================================================================
    std::vector<torch::jit::Value*> user_inputs = {
        b_weights,
        dataset_inputs_list,
        dataset_targets_list,
        loop_iter,
        loop_cond_in
    };

    // CORREZIONE: Inlineamento dei nodi del sottografo direttamente dentro il blocco interno del loop
    std::vector<torch::jit::Value*> uo = torch::jit::insertGraph(
        *main_graph,
        *user_step_subgraph,
        user_inputs
    );


    std::vector<c10::TypePtr> user_expected_types = { c10::TensorType::get(), c10::BoolType::get() };
    auto user_outputs = extractTupleResult(uo[0], main_graph.get(), 2, user_expected_types);
    torch::jit::Value* current_loss   = user_outputs[0];
    torch::jit::Value* next_loop_cond = user_outputs[1];

    // BLINDARE I TIPI DEL BLOCCO UTENTE:
    current_loss->setType(c10::TensorType::get());     // La loss deve essere un Tensor
    next_loop_cond->setType(c10::BoolType::get());    // La condizione deve essere un Bool


    // =========================================================================
    // PASSO 2: GESTIONE INFRASTRUTTURALE DI aten::grad (All'interno di body)
    // =========================================================================
    torch::jit::Node* outputs_list = main_graph->create(torch::jit::prim::ListConstruct, {current_loss});
    body->appendNode(outputs_list); // Inserimento esplicito nel blocco corrente del loop

    torch::jit::Node* grad_node = main_graph->create(torch::jit::aten::grad, {
        outputs_list->output(), b_weights, none_val, none_val, none_val, main_graph->insertConstant(false)
    });
    body->appendNode(grad_node); // Inserimento esplicito nel blocco corrente del loop
    torch::jit::Value* computed_gradients = grad_node->output();
    auto tensor_list_type = c10::ListType::create(c10::TensorType::get());
    computed_gradients->setType(tensor_list_type);
    // computed_gradients->setType(c10::ListType::create(c10::TensorType::get())); // Forza il tipo della lista di gradienti

    // =========================================================================
    // PASSO 3: INNESTO DEL SOTTOGRAFO DELL'OTTIMIZZATORE (All'interno di body)
    // =========================================================================
    std::vector<torch::jit::Value*> optimizer_inputs = {
        b_weights,
        computed_gradients,
        b_lr,
        b_m_list,
        b_v_list,
        loop_iter
    };

    std::vector<torch::jit::Value*> oo = torch::jit::insertGraph(
        *main_graph,
        *optimizer_subgraph,
        optimizer_inputs
    );
    // Specifichiamo che l'ottimizzatore restituisce: [Tensor[], float, Tensor[], Tensor[]]
    std::vector<c10::TypePtr> optimizer_expected_types = {
        tensor_list_type,       // next_weights_list
        c10::FloatType::get(),  // next_lr
        tensor_list_type,       // next_m_list
        tensor_list_type        // next_v_list
    };
    auto optimizer_outputs = extractTupleResult(oo[0], main_graph.get(), 4, optimizer_expected_types);

    torch::jit::Value* next_weights_list = optimizer_outputs[0];
    torch::jit::Value* next_lr           = optimizer_outputs[1];
    torch::jit::Value* next_m_list       = optimizer_outputs[2];
    torch::jit::Value* next_v_list       = optimizer_outputs[3];


    // =========================================================================
    // CHIUSURA REGISTRI (Dataflow allineato nel Loop)
    // =========================================================================
    // auto tensor_list_type = c10::ListType::create(c10::TensorType::get());
    next_weights_list->setType(tensor_list_type);
    next_lr->setType(c10::FloatType::get());
    next_m_list->setType(tensor_list_type);
    next_v_list->setType(tensor_list_type);
    // -------------------------------------------------------------------------
    // CHIUSURA REGISTRI DEL LOOP
    // -------------------------------------------------------------------------
    body->registerOutput(next_loop_cond);
    body->registerOutput(next_weights_list);
    body->registerOutput(next_lr);
    body->registerOutput(next_m_list);
    body->registerOutput(next_v_list);
    // -------------------------------------------------------------------------
    // CORREZIONE STRUTTURALE RADICE: Converte in Tupla singola per il salvataggio .pt
    // -------------------------------------------------------------------------
    torch::jit::Value* final_weights_out = loop_node->output(0);
    final_weights_out->setType(c10::ListType::create(c10::TensorType::get()));
    // torch::jit::Node* tuple_node = main_graph->create(torch::jit::prim::TupleConstruct, {final_weights_out});
    // main_graph->insertNode(tuple_node);
    main_graph->registerOutput(final_weights_out);
}
