//
// Created by gyankos on 03/10/26.
//

#ifndef METATENSOR_JITFLYWEIGHTDECORATOR_H
#define METATENSOR_JITFLYWEIGHTDECORATOR_H
#include <unordered_map>

#include "logds/tensormetadata/Metadata.h"

#include <torch/csrc/jit/ir/ir.h>

#include "logds/tensormetadata/LambdaAst.h"

// =============================================================================
// 1. BRIDGE IMPLEMENTOR INTERFACE (The Execution/Backend Dimension)
// =============================================================================
class JitExecutionBridge {
public:
    virtual ~JitExecutionBridge() = default;

    // Inietta i nodi operazionali all'interno del grafo intermedio di LibTorch
    virtual torch::jit::Value* compile_to_libtorch(
        std::shared_ptr<torch::jit::Graph>& graph,
        std::unordered_map<std::string, torch::jit::Value*>& jit_registry)  = 0;
};


class JitFlyweightDecorator : public TensorTyping, public JitExecutionBridge {
private:
    TypingNodePtr intrinsic_type_node; // L'informazione geometrico-relazionale di tipo pura

    // Stato Estrinseco Contestuale del Termine (Sradicate le vecchie stringhe grezze)
    LambdaAstPtr  extrinsic_lambda_ast_root;   // La radice del grafo logico-condizionale
    std::string extrinsic_disk_filepath;     // Il percorso fisico reale del file
    // float       extrinsic_scalar_value = 0.0f;

public:
    JitFlyweightDecorator(TypingNodePtr type_node)
        : intrinsic_type_node(std::move(type_node)) {}

    // Metodi di iniezione dello stato Flyweight
    void set_extrinsic_lambda_graph(LambdaAstPtr lambda_root) { extrinsic_lambda_ast_root = std::move(lambda_root); }
    void set_extrinsic_filepath(std::string path) { extrinsic_disk_filepath = std::move(path); }
    // void set_extrinsic_scalar(float val) { extrinsic_scalar_value = val; }

    TypingNodePtr get_intrinsic_type() const { return intrinsic_type_node; }
    LambdaAstPtr get_extrinsic_lambda() const { return extrinsic_lambda_ast_root; }
    std::string get_filepath() const { return extrinsic_disk_filepath; }
    // float get_scalar() const { return extrinsic_scalar_value; }

    torch::jit::Value* compile_to_libtorch(
        std::shared_ptr<torch::jit::Graph>& graph,
        std::unordered_map<std::string, torch::jit::Value*>& jit_registry)  override;
};



#endif //METATENSOR_JITFLYWEIGHTDECORATOR_H
