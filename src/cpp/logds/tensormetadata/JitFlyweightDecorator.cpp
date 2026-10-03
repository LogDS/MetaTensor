//
// Created by gyankos on 03/10/26.
//

#include "logds/tensormetadata/JitFlyweightDecorator.h"
// RISOLUTIVO: Forza Mach7 a NON registrare la macro globale corta "Match"
// per impedire il conflitto distruttivo con l'header typeid.h di LibTorch Core

#include <logds/tensormetadata/AllMachs7.h>

/*
 * This file is part of the MetaTensor distribution (https://github.com).
 * Copyright (c) 2026 Giacomo Bergami, PhD
 */

#include "logds/tensormetadata/Metadata.h"
#include "logds/tensormetadata/JitFlyweightDecorator.h"
#include <logds/tensormetadata/LambdaAst.h>
#include <stdexcept>
#include <algorithm>
#include <vector>


using namespace mch;

// Forward declaration dell'helper ricorsivo per la gerarchia parallela del LambdaAst
torch::jit::Value *compile_lambda_node_to_libtorch(
    const LambdaAstPtr &lambda_node,
    std::shared_ptr<torch::jit::Graph> &graph,
    std::unordered_map<std::string, torch::jit::Value *> &jit_registry,
    torch::jit::Value *hardware_cell_val);

// =============================================================================
// BACKEND DEL BRIDGE: EMISSIONE GRAFO COMPLETA (TensorTyping Hierarchy)
// =============================================================================
torch::jit::Value *JitFlyweightDecorator::compile_to_libtorch(
    std::shared_ptr<torch::jit::Graph> &graph,
    std::unordered_map<std::string, torch::jit::Value *> &jit_registry)  {
    // Se l'involucro intrinseco è nullo o è esso stesso un decoratore, effettua il downstream diretto
    if (!intrinsic_type_node) return nullptr;

    // Se l'intrinseco è un altro decoratore Flyweight, deleghiamo la chiamata per srotolare la catena
    if (auto nested_decorator = (JitFlyweightDecorator *)(intrinsic_type_node.get())) {
        return nested_decorator->compile_to_libtorch(graph, jit_registry);
    }

    // Registri di estrazione posizionale per le macro Match/Case di Mach7
    var<std::string> s, layer;
    var<DataType> dt;
    var<float> f1;
    var<size_t> axis, l_axis, r_axis;
    var<uint64_t> u1;
    var<TypingNodePtr> c_node, r_node;
    var<JitTensorMetadata> meta_data;
    //std::vector<std::variant<std::string, uint64_t>>
    var<std::vector<std::variant<std::string, uint64_t> > > v_axes;
    var<std::vector<size_t>> v_axes2;
    var<std::vector<TypingNodePtr> > v_params, v_body;
   var< DataType> target_type;
    TensorTyping *node = intrinsic_type_node.get();

    // Pattern Matching in-memory sul nucleo intrinseco pesante del Tipo
    Match(node) {
            // --- NODI FOGLIA, I/O E CARICAMENTO FILE SAFETENSORS ---
            Case(C<VariableNode>(s, meta_data)) {
                // Se la variabile è associata a un file su disco, il Flyweight decorator ne conserva il filepath
                if (!extrinsic_disk_filepath.empty()) {
                    torch::jit::Node *node = graph->create(
                        torch::jit::Symbol::fromQualString("metatensor::load_safetensors"), 1);
                    node->addInput(graph->insertConstant(extrinsic_disk_filepath));
                    node->addInput(graph->insertConstant(s)); // Nome del layer all'interno dell'archivio
                    graph->insertNode(node);
                    return node->output();
                }
                // Altrimenti viene registrata o recuperata come input formale del modulo di rete
                if (jit_registry.find(s) == jit_registry.end()) {
                    jit_registry[s] = graph->addInput(s);
                }
                return jit_registry[s];
            }

            Case(C<LoadSafeNode>(s, layer)) {
                // Fallback I/O se non catturato dal VariableNode: estrae i letterali dall'AST
                torch::jit::Node *node = graph->create(
                    torch::jit::Symbol::fromQualString("metatensor::load_safetensors"), 1);
                node->addInput(graph->insertConstant(s));
                node->addInput(graph->insertConstant(layer));
                graph->insertNode(node);
                return node->output();
            }

            Case(C<FromScalarNode>(f1, target_type)) {
                // Genera l'iniezione atomica di un valore float costante congelato nell'IR
                float f = f1;
                return graph->insertConstant(f);
            }

            // --- OPERATORI ALGEBRICI E CONTRAZIONI RELAZIONALI BINDING ---
            Case(C<MatMulNode>(c_node, r_node)) {
                torch::jit::Value *L = JitFlyweightDecorator(c_node).compile_to_libtorch(graph, jit_registry);
                torch::jit::Value *R = JitFlyweightDecorator(r_node).compile_to_libtorch(graph, jit_registry);

                torch::jit::Node *node = graph->create(torch::jit::Symbol::fromQualString("aten::matmul"), 1);
                node->addInput(L);
                node->addInput(R);
                graph->insertNode(node);
                return node->output();
            }

            Case(C<AddNode>(c_node, r_node)) {
                torch::jit::Value *L = JitFlyweightDecorator(c_node).compile_to_libtorch(graph, jit_registry);
                torch::jit::Value *R = JitFlyweightDecorator(r_node).compile_to_libtorch(graph, jit_registry);

                torch::jit::Node *node = graph->create(torch::jit::Symbol::fromQualString("aten::add"), 1);
                node->addInput(L);
                node->addInput(R);
                node->addInput(graph->insertConstant(1)); // Alpha scalare predefinito dalle ABI di ATen
                graph->insertNode(node);
                return node->output();
            }

            Case(C<SubNode>(c_node, r_node)) {
                torch::jit::Value *L = JitFlyweightDecorator(c_node).compile_to_libtorch(graph, jit_registry);
                torch::jit::Value *R = JitFlyweightDecorator(r_node).compile_to_libtorch(graph, jit_registry);

                torch::jit::Node *node = graph->create(torch::jit::Symbol::fromQualString("aten::sub"), 1);
                node->addInput(L);
                node->addInput(R);
                node->addInput(graph->insertConstant(1));
                graph->insertNode(node);
                return node->output();
            }

            Case(C<CrossNode>(c_node, r_node)) {
                torch::jit::Value *L = JitFlyweightDecorator(c_node).compile_to_libtorch(graph, jit_registry);
                torch::jit::Value *R = JitFlyweightDecorator(r_node).compile_to_libtorch(graph, jit_registry);

                torch::jit::Node *node = graph->create(torch::jit::Symbol::fromQualString("aten::cross"), 1);
                node->addInput(L);
                node->addInput(R);
                node->addInput(graph->insertConstant(0)); // Asse spaziale predefinito 0 per cross 3D
                graph->insertNode(node);
                return node->output();
            }

            Case(C<HadamardNode>(c_node, r_node)) {
                torch::jit::Value *L = JitFlyweightDecorator(c_node).compile_to_libtorch(graph, jit_registry);
                torch::jit::Value *R = JitFlyweightDecorator(r_node).compile_to_libtorch(graph, jit_registry);

                torch::jit::Node *node = graph->create(torch::jit::Symbol::fromQualString("aten::mul"), 1);
                node->addInput(L);
                node->addInput(R);
                graph->insertNode(node);
                return node->output();
            }

            Case(C<ThetaJoinNode>(c_node, r_node)) {
                torch::jit::Value *L = JitFlyweightDecorator(c_node).compile_to_libtorch(graph, jit_registry);
                torch::jit::Value *R = JitFlyweightDecorator(r_node).compile_to_libtorch(graph, jit_registry);

                torch::jit::Node *node = graph->create(
                    torch::jit::Symbol::fromQualString("metatensor::tensor_theta_join"), 1);
                node->addInput(L);
                node->addInput(R);
                graph->insertNode(node);
                return node->output();
            }

            // --- OPERATORI UNARI E RIDUZIONI D'ASSE ---
            Case(C<ApplyUnaryNode>(c_node, s)) {
                torch::jit::Value *child_val = JitFlyweightDecorator(c_node).compile_to_libtorch(graph, jit_registry);
                std::string aten_symbol = "aten::" + s;
                std::transform(aten_symbol.begin(), aten_symbol.end(), aten_symbol.begin(), ::tolower);

                torch::jit::Node *node = graph->create(torch::jit::Symbol::fromQualString(aten_symbol), 1);
                node->addInput(child_val);
                graph->insertNode(node);
                return node->output();
            }

            Case(C<ReduceSumNode>(c_node, axis)) {
                torch::jit::Value *child_val = JitFlyweightDecorator(c_node).compile_to_libtorch(graph, jit_registry);

                torch::jit::Node *node = graph->create(torch::jit::Symbol::fromQualString("aten::sum"), 1);
                node->addInput(child_val);
                node->addInput(graph->insertConstant(static_cast<int64_t>(axis)));
                node->addInput(graph->insertConstant(false)); // keepdim = false
                graph->insertNode(node);
                return node->output();
            }

            Case(C<ReduceAllNode>(c_node)) {
                torch::jit::Value *child_val = JitFlyweightDecorator(c_node).compile_to_libtorch(graph, jit_registry);

                torch::jit::Node *node = graph->create(torch::jit::Symbol::fromQualString("aten::sum"), 1);
                graph->insertNode(node);
                return node->output();
            }

            Case(C<GatherNdNode>(c_node, r_node)) {
                torch::jit::Value *S = JitFlyweightDecorator(c_node).compile_to_libtorch(graph, jit_registry);
                torch::jit::Value *I = JitFlyweightDecorator(r_node).compile_to_libtorch(graph, jit_registry);

                torch::jit::Node *node = graph->create(torch::jit::Symbol::fromQualString("metatensor::gather_nd"), 1);
                node->addInput(S);
                node->addInput(I);
                graph->insertNode(node);
                return node->output();
            }

            Case(C<PermuteAxesNode>(c_node, v_axes)) {
                torch::jit::Value *child_val = JitFlyweightDecorator(c_node).compile_to_libtorch(graph, jit_registry);

                std::vector<int64_t> concrete_axes;
                for (const auto &ax: v_axes) {
                    if (std::holds_alternative<uint64_t>(ax)) {
                        concrete_axes.push_back(std::get<uint64_t>(ax));
                    } else {
                        // Fallback se l'asse è rimasto puramente simbolico a runtime: usiamo un indice di default
                        concrete_axes.push_back(0);
                    }
                }

                torch::jit::Node *node = graph->create(torch::jit::Symbol::fromQualString("aten::permute"), 1);
                node->addInput(child_val);
                node->addInput(graph->insertConstant(concrete_axes));
                graph->insertNode(node);
                return node->output();
            }
            // =============================================================================
            // ∃ & ∀ QUANTIFICATORI: COMPILAZIONE INTEGRATA DELL'AST DELLE LAMBDA
            // =============================================================================
            Case(C<ExistentialQuantifierNode>(c_node, v_axes2)) {
                // 1. Abbassamento hardware del tensore d'ingresso
                torch::jit::Value *source_tensor_val = JitFlyweightDecorator(c_node).compile_to_libtorch(
                    graph, jit_registry);
                // 2. RISOLUTIVO: Estraiamo l'AST della lambda memorizzato nello stato estrinseco del Flyweight
                if (!extrinsic_lambda_ast_root) {
                    throw std::runtime_error(
                        "[JIT_COMPILE_FATAL] Nodo Esistenziale orfano: Manca la radice LambdaAst.");
                }
                // Invochiamo il compilatore parallelo ricorsivo del LambdaAst immettendo il valore della cella corrente
                torch::jit::Value *boolean_mask_val = compile_lambda_node_to_libtorch(
                    extrinsic_lambda_ast_root, graph, jit_registry, source_tensor_val
                );
                // 3. Riduzione bitwise hardware tramite l'operatore logico aten::any
                torch::jit::Value *current_reduction_val = boolean_mask_val;
                for (const auto &ax: v_axes) {
                    if (std::holds_alternative<uint64_t>(ax)) {
                        int64_t concrete_axis = std::get<uint64_t>(ax);
                        torch::jit::Node *any_node = graph->create(torch::jit::Symbol::fromQualString("aten::any"), 1);
                        any_node->addInput(current_reduction_val);
                        any_node->addInput(graph->insertConstant(concrete_axis));
                        any_node->addInput(graph->insertConstant(false));
                        graph->insertNode(any_node);
                        current_reduction_val = any_node->output();
                    }
                }
                // Cast finale del binario booleano i1 nel tipo float f32 nativo
                torch::jit::Node *cast_node = graph->create(torch::jit::Symbol::fromQualString("aten::to"), 1);
                cast_node->addInput(current_reduction_val);
                cast_node->addInput(graph->insertConstant(static_cast<int64_t>(at::kFloat)));
                cast_node->addInput(graph->insertConstant(false));
                cast_node->addInput(graph->insertConstant(false));
                graph->insertNode(cast_node);
                return cast_node->output();
            }
            // --- ORCHESTRAZIONE DEL NASTRO DELLE EPOCCHE (ActiveEpochBlockNode) ---
        Case(C<EpochIterationNode>(u1, f1, v_params, v_body)) {
                torch::jit::Value* last_statement_val = nullptr;

                // 1. Eseguiamo l'unrolling lineare del ciclo delle epoche del nastro nel grafo JIT
                for (uint64_t step = 0; step < u1; ++step) {
                    for (const auto& statement_ptr : v_body) {
                        last_statement_val = JitFlyweightDecorator(statement_ptr).compile_to_libtorch(graph, jit_registry);
                    }

                    // 2. RISOLUTIVO: Per ciascun parametro (peso/bias) sotto osservazione inserito nel nastro,
                    // iniettiamo l'istruzione di ottimizzazione applicando il tasso di apprendimento corretto f1
                    for (const auto& param_node_ptr : v_params) {
                        // Estraiamo il nodo variabile reale per recuperarne l'identificatore stringa nel registro JIT
                        if (auto* var_node = dynamic_cast<const VariableNode*>(param_node_ptr.get())) {
                            std::string target_weight_name = var_node->name;

                            if (jit_registry.find(target_weight_name) != jit_registry.end()) {
                                torch::jit::Node* step_node = graph->create(
                                    torch::jit::Symbol::fromQualString("metatensor::apply_optimization_step"), 0);
                                float f = f1;

                                step_node->addInput(jit_registry[target_weight_name]); // Inietta il Value* del peso reale
                                step_node->addInput(graph->insertConstant(f));                 // Inietta il base_learning_rate float
                                step_node->addInput(graph->insertConstant(static_cast<int64_t>(((EpochIterationNode*)node)->optimizer_type))); // Inietta l'ottimizzatore (SGD/Adam/etc)
                                graph->insertNode(step_node);
                            }
                        }
                    }
                }
                return last_statement_val;
            }
        // --- PUNTO DI INGRESSO E TERMINAZIONE DEL GRAFO (ProgramRootNode) ---
        // u1 cattura la lista 'active_configurations' (direttive use device, optimizer, ecc.)
        // v_body cattura la sequenza sequenziale di statement ed epoch_loop del DSL
        Case(C<ProgramRootNode>(v_params, v_body)) {
                torch::jit::Value* final_return_val = nullptr;

                // Compiliamo in sequenza ogni statement del programma nell'IR di LibTorch
                for (const auto& stmt : v_body) {
                    final_return_val = JitFlyweightDecorator(stmt).compile_to_libtorch(graph, jit_registry);
                }

                // RISOLUTIVO: Se il grafo ha prodotto un valore valido, lo marchiamo come output formale
                if (final_return_val) {
                    graph->registerOutput(final_return_val);
                }
                return final_return_val;
            }

            Otherwise() {
                    throw std::runtime_error(
                        "[JIT_BRIDGE_FATAL] Nodo di TensorTyping non mappabile nell'IR di LibTorch.");
                }
            }
    EndMatch
    return nullptr;
}


// =============================================================================
// COMPILATORE RICORSIVO PARALLELO DEL LAMBDA_AST (In-Memory JIT)
// =============================================================================
torch::jit::Value* compile_lambda_node_to_libtorch(
    const LambdaAstPtr& lambda_node,
    std::shared_ptr<torch::jit::Graph>& graph,
    std::unordered_map<std::string, torch::jit::Value*>& jit_registry,
    torch::jit::Value* hardware_cell_val) {

    if (!lambda_node) return nullptr;

    // Registri di estrazione posizionale fortemente tipizzati per Mach7
    var<std::string> var_id;
    var<float> f_lit;
    var<FloatBinaryOp> b_op;
    var<FloatUnaryOp> u_op;
    var<FloatCompOp> c_op;
    var<IntrinsicPred> i_pred;

    var<FloatExprPtr> f_left, f_right, f_child;
    var<BoolCondPtr> b_left, b_right, b_child;

    // Pattern Matching di Mach7 applicato alla gerarchia LambdaAst indipendente
    Match(lambda_node) {
        // --- NODI FOGLIA / TERMINALI DEL PREDICATO ---
        Case(C<CellTerminalNode>()) {
            return hardware_cell_val;
        }

        Case(C<FloatVariableIdNode>(var_id)) {
            if (jit_registry.find(var_id) == jit_registry.end()) {
                throw std::runtime_error("[JIT_LAMBDA_ERROR] Variabile float orfana nella lambda: " + var_id);
            }
            return jit_registry[var_id];
        }

        Case(C<FloatValueLiteralNode>(f_lit)) {
            float f = f_lit;
            return graph->insertConstant(f);
        }

        Case(C<IntrinsicPredicateNode>(i_pred)) {
            if (i_pred == IntrinsicPred::ISNAN) {
                torch::jit::Node* node = graph->create(torch::jit::Symbol::fromQualString("aten::isnan"), 1);
                node->addInput(hardware_cell_val);
                graph->insertNode(node);
                return node->output();
            }
            throw std::runtime_error("[JIT_LAMBDA_ERROR] Predicato intrinseco non supportato nell'IR.");
        }

        // --- OPERATORI ARITMETICI SCALARI ---
        Case(C<FloatBinaryOpNode>(f_left, f_right, b_op)) {
            torch::jit::Value* L = compile_lambda_node_to_libtorch(f_left, graph, jit_registry, hardware_cell_val);
            torch::jit::Value* R = compile_lambda_node_to_libtorch(f_right, graph, jit_registry, hardware_cell_val);
            torch::jit::Symbol sym;

            if (b_op == FloatBinaryOp::ADD) sym = torch::jit::Symbol::fromQualString("aten::add");
            else if (b_op == FloatBinaryOp::SUB) sym = torch::jit::Symbol::fromQualString("aten::sub");
            else if (b_op == FloatBinaryOp::DIV) sym = torch::jit::Symbol::fromQualString("aten::div");

            torch::jit::Node* node = graph->create(sym, 1);
            node->addInput(L); node->addInput(R);
            if (b_op != FloatBinaryOp::DIV) node->addInput(graph->insertConstant(1));
            graph->insertNode(node);
            return node->output();
        }

        Case(C<FloatUnaryOpNode>(f_child, u_op)) {
            torch::jit::Value* child = compile_lambda_node_to_libtorch(f_child, graph, jit_registry, hardware_cell_val);
            if (u_op == FloatUnaryOp::ABS) {
                torch::jit::Node* node = graph->create(torch::jit::Symbol::fromQualString("aten::abs"), 1);
                node->addInput(child);
                graph->insertNode(node);
                return node->output();
            }
            return nullptr;
        }

        // --- COMPARAZIONI RELAZIONALI UNIVERSALI ---
        Case(C<FloatComparisonNode>(f_left, f_right, c_op)) {
            torch::jit::Value* L = compile_lambda_node_to_libtorch(f_left, graph, jit_registry, hardware_cell_val);
            torch::jit::Value* R = compile_lambda_node_to_libtorch(f_right, graph, jit_registry, hardware_cell_val);
            torch::jit::Symbol sym;

            switch (c_op) {
                case FloatCompOp::GT: sym = torch::jit::Symbol::fromQualString("aten::gt"); break;
                case FloatCompOp::LT: sym = torch::jit::Symbol::fromQualString("aten::lt"); break;
                case FloatCompOp::GE: sym = torch::jit::Symbol::fromQualString("aten::ge"); break;
                case FloatCompOp::LE: sym = torch::jit::Symbol::fromQualString("aten::le"); break;
                case FloatCompOp::EQ: sym = torch::jit::Symbol::fromQualString("aten::eq"); break;
                case FloatCompOp::NE: sym = torch::jit::Symbol::fromQualString("aten::ne"); break;
            }

            torch::jit::Node* node = graph->create(sym, 1);
            node->addInput(L); node->addInput(R);
            graph->insertNode(node);
            return node->output();
        }

        // --- CONNETTIVI LOGICI BOOLEANI (AND / OR / NOT) ---
        Case(C<LogicalAndNode>(b_left, b_right)) {
            torch::jit::Value* L = compile_lambda_node_to_libtorch(b_left, graph, jit_registry, hardware_cell_val);
            torch::jit::Value* R = compile_lambda_node_to_libtorch(b_right, graph, jit_registry, hardware_cell_val);

            torch::jit::Node* node = graph->create(torch::jit::Symbol::fromQualString("aten::__and__"), 1);
            node->addInput(L); node->addInput(R);
            graph->insertNode(node);
            return node->output();
        }

        Case(C<LogicalOrNode>(b_left, b_right)) {
            torch::jit::Value* L = compile_lambda_node_to_libtorch(b_left, graph, jit_registry, hardware_cell_val);
            torch::jit::Value* R = compile_lambda_node_to_libtorch(b_right, graph, jit_registry, hardware_cell_val);

            torch::jit::Node* node = graph->create(torch::jit::Symbol::fromQualString("aten::__or__"), 1);
            node->addInput(L); node->addInput(R);
            graph->insertNode(node);
            return node->output();
        }

        Case(C<LogicalNotNode>(b_child)) {
            torch::jit::Value* child = compile_lambda_node_to_libtorch(b_child, graph, jit_registry, hardware_cell_val);

            torch::jit::Node* node = graph->create(torch::jit::Symbol::fromQualString("aten::__not__"), 1);
            node->addInput(child);
            graph->insertNode(node);
            return node->output();
        }

        Otherwise() {
            throw std::runtime_error("[JIT_LAMBDA_FATAL] Nodo di LambdaAst non implementato o non mappabile nell'IR di LibTorch.");
        }
    }
    EndMatch
    return nullptr;
}

//
// // =============================================================================
// // COMPILATORE RICORSIVO PARALLELO DEL NUOVO LAMBDA_AST (In-Memory JIT)
// // =============================================================================
// torch::jit::Value *compile_lambda_node_to_libtorch(
//     const LambdaAstPtr &lambda_node,
//     std::shared_ptr<torch::jit::Graph> &graph,
//     std::unordered_map<std::string, torch::jit::Value *> &jit_registry,
//     torch::jit::Value *hardware_cell_val) {
//     if (!lambda_node) return nullptr;
//     // Registri di estrazione posizionale per la gerarchia LambdaAst
//     var<std::string> var_id;
//     var<float> f_lit;
//     var<FloatBinaryOp> b_op;
//     var<FloatUnaryOp> u_op;
//     var<FloatCompOp> c_op;
//     var<IntrinsicPred> i_pred;
//
//     // Specifichiamo i tipi esatati per i puntatori smart dell'AST parallelo
//     var<FloatExprPtr> f_left, f_right, f_child;
//     var<BoolCondPtr> b_left, b_right, b_child;
//     // Pattern Matching di Mach7 applicato alla gerarchia LambdaAst indipendenti
//     Match(*lambda_node) {
//             // --- NODI FOGLIA / TERMINALI DEL PREDICATO ---
//             Case(C()) {
//                 // Sostituisce la parola chiave 'cell' con il registro del valore del tensore d'ingresso
//                 return hardware_cell_val;
//             }
//             Case(C(var_id)) {
//                 if (jit_registry.find(var_id) == jit_registry.end()) {
//                     throw std::runtime_error("[JIT_LAMBDA_ERROR] Variabile float orfana nella lambda: " + var_id);
//                 }
//                 return jit_registry[var_id];
//             }
//             Case(C(f_lit)) {
//                 return graph->insertConstant(f_lit);
//             }
//             Case(C(i_pred)) {
//                 // Gestione dei predicati impliciti (es: isnan)
//                 if (i_pred == IntrinsicPred::ISNAN) {
//                     torch::jit::Node *node = graph->create(torch::jit::Symbol::fromQualString("aten::isnan"), 1);
//                     node->addInput(hardware_cell_val);
//                     graph->insertNode(node);
//                     return node->output();
//                 }
//                 throw std::runtime_error("[JIT_LAMBDA_ERROR] Predicato intrinseco non supportato nell'IR.");
//             }
//             // --- OPERATORI ARITMETICI SCALARI ---
//             Case(C(f_left, f_right, b_op)) {
//                 torch::jit::Value *L = compile_lambda_node_to_libtorch(f_left, graph, jit_registry, hardware_cell_val);
//                 torch::jit::Value *R = compile_lambda_node_to_libtorch(f_right, graph, jit_registry, hardware_cell_val);
//                 torch::jit::Symbol sym;
//                 if (b_op == FloatBinaryOp::ADD) sym = torch::jit::Symbol::fromQualString("aten::add");
//                 else if (b_op == FloatBinaryOp::SUB) sym = torch::jit::Symbol::fromQualString("aten::sub");
//                 else if (b_op == FloatBinaryOp::DIV) sym = torch::jit::Symbol::fromQualString("aten::div");
//                 torch::jit::Node *node = graph->create(sym, 1);
//                 node->addInput(L);
//                 node->addInput(R);
//                 if (b_op != FloatBinaryOp::DIV) node->addInput(graph->insertConstant(1)); // Alpha per add/sub
//                 graph->insertNode(node);
//                 return node->output();
//             }
//             Case(C(f_child, u_op)) {
//                 torch::jit::Value *child = compile_lambda_node_to_libtorch(
//                     f_child, graph, jit_registry, hardware_cell_val);
//                 if (u_op == FloatUnaryOp::ABS) {
//                     torch::jit::Node *node = graph->create(torch::jit::Symbol::fromQualString("aten::abs"), 1);
//                     node->addInput(child);
//                     graph->insertNode(node);
//                     return node->output();
//                 }
//                 return nullptr;
//             }
//             // --- COMPARAZIONI RELAZIONALI UNIVERSALI ---
//             Case(C(f_left, f_right, c_op)) {
//                 torch::jit::Value *L = compile_lambda_node_to_libtorch(f_left, graph, jit_registry, hardware_cell_val);
//                 torch::jit::Value *R = compile_lambda_node_to_libtorch(f_right, graph, jit_registry, hardware_cell_val);
//                 torch::jit::Symbol sym;
//                 switch (c_op) {
//                     case FloatCompOp::GT: sym = torch::jit::Symbol::fromQualString("aten::gt");
//                         break;
//                     case FloatCompOp::LT: sym = torch::jit::Symbol::fromQualString("aten::lt");
//                         break;
//                     case FloatCompOp::GE: sym = torch::jit::Symbol::fromQualString("aten::ge");
//                         break;
//                     case FloatCompOp::LE: sym = torch::jit::Symbol::fromQualString("aten::le");
//                         break;
//                     case FloatCompOp::EQ: sym = torch::jit::Symbol::fromQualString("aten::eq");
//                         break;
//                     case FloatCompOp::NE: sym = torch::jit::Symbol::fromQualString("aten::ne");
//                         break;
//                 }
//                 torch::jit::Node *node = graph->create(sym, 1);
//                 node->addInput(L);
//                 node->addInput(R);
//                 graph->insertNode(node);
//                 return node->output();
//             }
//             // --- CONNETTIVI LOGICI BOOLEANI (AND / OR / NOT) ---
//             Case(C(b_left, b_right)) {
//                 torch::jit::Value *L = compile_lambda_node_to_libtorch(b_left, graph, jit_registry, hardware_cell_val);
//                 torch::jit::Value *R = compile_lambda_node_to_libtorch(b_right, graph, jit_registry, hardware_cell_val);
//                 torch::jit::Node *node = graph->create(torch::jit::Symbol::fromQualString("aten::and"), 1);
//                 node->addInput(L);
//                 node->addInput(R);
//                 graph->insertNode(node);
//                 return node->output();
//             }
//             Case(C(b_left, b_right)) {
//                 torch::jit::Value *L = compile_lambda_node_to_libtorch(b_left, graph, jit_registry, hardware_cell_val);
//                 torch::jit::Value *R = compile_lambda_node_to_libtorch(b_right, graph, jit_registry, hardware_cell_val);
//                 torch::jit::Node *node = graph->create(torch::jit::Symbol::fromQualString("aten::or"), 1);
//                 node->addInput(L);
//                 node->addInput(R);
//                 graph->insertNode(node);
//                 return node->output();
//             }
//             Case(C(b_child)) {
//                 torch::jit::Value *child = compile_lambda_node_to_libtorch(
//                     b_child, graph, jit_registry, hardware_cell_val);
//                 torch::jit::Node *node = graph->create(torch::jit::Symbol::fromQualString("aten::not"), 1);
//                 node->addInput(child);
//                 graph->insertNode(node);
//                 return node->output();
//             }
//             Otherwise() {
//                     throw std::runtime_error(
//                         "[JIT_LAMBDA_FATAL] Nodo di LambdaAst non implementato o non mappabile nell'IR di LibTorch.");
//                 }
//             }
//     EndMatch
// };