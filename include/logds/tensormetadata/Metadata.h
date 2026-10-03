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
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef TENSORLIBRARY_JITASTNODES_H
#define TENSORLIBRARY_JITASTNODES_H




#include <logds/metatensor/StorageLayout.h>
#include <vector>
#include <string>
#include <variant>
#include <optional>
#include <stdexcept>
#include <algorithm>
#include <memory>
#include <string>
#include <vector>



typedef enum {
    TYPE_FLOAT,
    TYPE_DOUBLE,
    TYPE_INT32
} DataType;

typedef struct {
    DataType data_type;
    StorageLayout layout;
    size_t rank;
    std::vector<std::variant<std::string, uint64_t> > shape;
    bool is_valid;
} JitTensorMetadata;


#include <string>
#include <vector>
#include <memory>

// Classe base polimorfa abilitata per lo shared_from_this se necessario
struct TensorTyping {
    virtual ~TensorTyping() = default;
};

// Alias di produzione per eliminare i memory leak nell'albero ANTLR4
using TypingNodePtr = std::shared_ptr< TensorTyping>;

// Foglie dell'Albero
struct VariableNode : TensorTyping {
    std::string name;
    JitTensorMetadata metadata;
    VariableNode(std::string n, JitTensorMetadata m) : name(n), metadata(m) {}
};

struct LoadSafeNode : TensorTyping {
    std::string filepath;
    std::string layer;
    LoadSafeNode(std::string f, std::string l) : filepath(f), layer(l) {}
};

struct FromScalarNode : TensorTyping {
    float value;
    DataType target_type;
    FromScalarNode(float v, DataType dt) : value(v), target_type(dt) {}
};

// Nodo per il dumping dei pesi (es: dump_safetensors)
struct DumpSafeNode : TensorTyping {
    TypingNodePtr tensor_node; // Il sotto-nodo da serializzare (es: W)
    std::string filepath;
    std::string tensor_name;

    DumpSafeNode(TypingNodePtr t, std::string f, std::string n)
        : tensor_node(std::move(t)), filepath(f), tensor_name(n) {}
};

// Nodo per l'esportazione multi-output (es: export(W, Prediction);)
struct ExportStatementNode : TensorTyping {
    std::vector<TypingNodePtr> output_nodes; // La collezione dei molteplici output estratti in tupla

    ExportStatementNode(std::vector<TypingNodePtr> nodes)
        : output_nodes(std::move(nodes)) {}
};

// Operatori Binari Element-wise e Relazionali (RAII Compliant)
struct AddNode       : TensorTyping { TypingNodePtr left; TypingNodePtr right; AddNode(TypingNodePtr l, TypingNodePtr r) : left(l), right(r) {} };
struct SubNode       : TensorTyping { TypingNodePtr left; TypingNodePtr right; SubNode(TypingNodePtr l, TypingNodePtr r) : left(l), right(r) {} };
struct MatMulNode    : TensorTyping { TypingNodePtr left; TypingNodePtr right; MatMulNode(TypingNodePtr l, TypingNodePtr r) : left(l), right(r) {} };
struct CrossNode     : TensorTyping { TypingNodePtr left; TypingNodePtr right; CrossNode(TypingNodePtr l, TypingNodePtr r) : left(l), right(r) {} };
struct HadamardNode  : TensorTyping { TypingNodePtr left; TypingNodePtr right; HadamardNode(TypingNodePtr l, TypingNodePtr r) : left(l), right(r) {} };
struct ThetaJoinNode : TensorTyping { TypingNodePtr left; TypingNodePtr right; ThetaJoinNode(TypingNodePtr l, TypingNodePtr r) : left(l), right(r) {} };
struct ContractionNode : TensorTyping {
    TypingNodePtr left; TypingNodePtr right; size_t l_axis; size_t r_axis;
    ContractionNode(TypingNodePtr l, TypingNodePtr r, size_t la, size_t ra) : left(l), right(r), l_axis(la), r_axis(ra) {}
};

// Operatori Unari e Slicing
struct ApplyUnaryNode : TensorTyping { TypingNodePtr child; std::string op_name; ApplyUnaryNode(TypingNodePtr c, std::string op) : child(c), op_name(op) {} };
struct ReduceSumNode  : TensorTyping { TypingNodePtr child; size_t axis; ReduceSumNode(TypingNodePtr c, size_t a) : child(c), axis(a) {} };
struct ReduceAllNode  : TensorTyping { TypingNodePtr child; ReduceAllNode(TypingNodePtr c) : child(c) {} };
struct CastingNode    : TensorTyping { TypingNodePtr child; DataType target_type; CastingNode(TypingNodePtr c, DataType dt) : child(c), target_type(dt) {} };
struct PrimitiveFloatExtractNode : TensorTyping { TypingNodePtr child; PrimitiveFloatExtractNode(TypingNodePtr c) : child(c) {} };
struct SliceTensorNode : TensorTyping {
    TypingNodePtr child; size_t axis; uint64_t start; uint64_t end;
    SliceTensorNode(TypingNodePtr c, size_t ax, uint64_t s, uint64_t e) : child(c), axis(ax), start(s), end(e) {}
};
struct SqueezeAxesNode : TensorTyping { TypingNodePtr child; size_t axis; SqueezeAxesNode(TypingNodePtr c, size_t ax) : child(c), axis(ax) {} };
struct SegmentSumNode  : TensorTyping { TypingNodePtr child; uint64_t num_segments; SegmentSumNode(TypingNodePtr c, uint64_t ns) : child(c), num_segments(ns) {} };
struct OneHotNode      : TensorTyping { TypingNodePtr child; uint64_t depth; OneHotNode(TypingNodePtr c, uint64_t d) : child(c), depth(d) {} };
struct GatherNdNode    : TensorTyping { TypingNodePtr source; TypingNodePtr indices; GatherNdNode(TypingNodePtr s, TypingNodePtr i) : source(s), indices(i) {} };
struct WhereNode       : TensorTyping { TypingNodePtr child; float off_value; WhereNode(TypingNodePtr c, float ov) : child(c), off_value(ov) {} };
struct BracketsNode    : TensorTyping { TypingNodePtr child; std::vector<uint64_t> coords; BracketsNode(TypingNodePtr c, std::vector<uint64_t> cc) : child(c), coords(cc) {} };
struct PermuteAxesNode : TensorTyping {
    TypingNodePtr child;
    std::vector<std::variant<std::string, uint64_t>> axes;
    PermuteAxesNode(TypingNodePtr c, std::vector<std::variant<std::string, uint64_t>> ax) : child(c), axes(ax) {}
};

// Operatori Funzionali Complessi con Predicati
struct ExistentialQuantifierNode : TensorTyping { TypingNodePtr child; std::vector<size_t> reduce_axes; ExistentialQuantifierNode(TypingNodePtr c, std::vector<size_t> ra) : child(c), reduce_axes(ra) {} };
struct UniversalQuantifierNode   : TensorTyping { TypingNodePtr child; std::vector<size_t> reduce_axes; UniversalQuantifierNode(TypingNodePtr c, std::vector<size_t> ra) : child(c), reduce_axes(ra) {} };
struct RelationalAggregateNode   : TensorTyping { TypingNodePtr child; std::vector<size_t> retained_axes; RelationalAggregateNode(TypingNodePtr c, std::vector<size_t> ra) : child(c), retained_axes(ra) {} };

// Il Nastro Parametrico dell'Epoch Iteration (RAII Memory Sandbox)
struct EpochIterationNode : TensorTyping {
    uint64_t epochs;
    float base_learning_rate;
    int64_t optimizer_type;
    std::vector<TypingNodePtr> gradient_parameters;
    std::vector<TypingNodePtr> computational_statements;

    EpochIterationNode(uint64_t ep, float lr, int64_t opt, std::vector<TypingNodePtr> params, std::vector<TypingNodePtr> body)
        : epochs(ep), base_learning_rate(lr), optimizer_type(opt), gradient_parameters(params), computational_statements(body) {}
};

// Nodo Programma Radice Completo
struct ProgramRootNode : TensorTyping {
    std::vector<TypingNodePtr> configurations;
    std::vector<TypingNodePtr> execution_graph;
    ProgramRootNode(std::vector<TypingNodePtr> c, std::vector<TypingNodePtr> e) : configurations(c), execution_graph(e) {}
};


#endif //TENSORLIBRARY_JITASTNODES_H
