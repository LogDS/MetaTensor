//
// Created by gyankos on 02/10/26.
//

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






// Classe base polimorfa per i nodi del grafo
struct AstNode {
    virtual ~AstNode() = default;
};

// Foglie del Grafo
struct VariableNode : AstNode {
    std::string name;
    JitTensorMetadata metadata;
    VariableNode(std::string n, JitTensorMetadata m) : name(n), metadata(m) {}
};

struct LoadSafeNode : AstNode {
    std::string filepath;
    std::string layer;
    LoadSafeNode(std::string f, std::string l) : filepath(f), layer(l) {}
};

struct FromScalarNode : AstNode {
    float value;
    DataType target_type;
    FromScalarNode(float v, DataType dt) : value(v), target_type(dt) {}
};

// Operatori Binari Element-wise e Relazionali
struct AddNode       : AstNode { AstNode* left; AstNode* right; AddNode(AstNode* l, AstNode* r) : left(l), right(r) {} };
struct SubNode       : AstNode { AstNode* left; AstNode* right; SubNode(AstNode* l, AstNode* r) : left(l), right(r) {} };
struct MatMulNode    : AstNode { AstNode* left; AstNode* right; MatMulNode(AstNode* l, AstNode* r) : left(l), right(r) {} };
struct CrossNode     : AstNode { AstNode* left; AstNode* right; CrossNode(AstNode* l, AstNode* r) : left(l), right(r) {} };
struct HadamardNode  : AstNode { AstNode* left; AstNode* right; HadamardNode(AstNode* l, AstNode* r) : left(l), right(r) {} };
struct ThetaJoinNode : AstNode { AstNode* left; AstNode* right; ThetaJoinNode(AstNode* l, AstNode* r) : left(l), right(r) {} };
struct ContractionNode : AstNode { AstNode* left; AstNode* right; size_t l_axis; size_t r_axis; ContractionNode(AstNode* l, AstNode* r, size_t la, size_t ra) : left(l), right(r), l_axis(la), r_axis(ra) {} };

// Operatori Unari e Slicing
struct ApplyUnaryNode : AstNode { AstNode* child; std::string op_name; ApplyUnaryNode(AstNode* c, std::string op) : child(c), op_name(op) {} };
struct ReduceSumNode  : AstNode { AstNode* child; size_t axis; ReduceSumNode(AstNode* c, size_t a) : child(c), axis(a) {} };
struct ReduceAllNode  : AstNode { AstNode* child; ReduceAllNode(AstNode* c) : child(c) {} };
struct CastingNode    : AstNode { AstNode* child; DataType target_type; CastingNode(AstNode* c, DataType dt) : child(c), target_type(dt) {} };
struct PrimitiveFloatExtractNode : AstNode { AstNode* child; PrimitiveFloatExtractNode(AstNode* c) : child(c) {} };
struct SliceTensorNode : AstNode { AstNode* child; size_t axis; uint64_t start; uint64_t end; SliceTensorNode(AstNode* c, size_t ax, uint64_t s, uint64_t e) : child(c), axis(ax), start(s), end(e) {} };
struct SqueezeAxesNode : AstNode { AstNode* child; size_t axis; SqueezeAxesNode(AstNode* c, size_t ax) : child(c), axis(ax) {} };
struct SegmentSumNode  : AstNode { AstNode* child; uint64_t num_segments; SegmentSumNode(AstNode* c, uint64_t ns) : child(c), num_segments(ns) {} };
struct OneHotNode      : AstNode { AstNode* child; uint64_t depth; OneHotNode(AstNode* c, uint64_t d) : child(c), depth(d) {} };
struct GatherNdNode    : AstNode { AstNode* source; AstNode* indices; GatherNdNode(AstNode* s, AstNode* i) : source(s), indices(i) {} };
struct WhereNode       : AstNode { AstNode* child; float off_value; WhereNode(AstNode* c, float ov) : child(c), off_value(ov) {} };
struct BracketsNode    : AstNode { AstNode* child; std::vector<uint64_t> coords; BracketsNode(AstNode* c, std::vector<uint64_t> cc) : child(c), coords(cc) {} };

// Operatori Funzionali Complessi con Predicati
struct ExistentialQuantifierNode : AstNode { AstNode* child; std::vector<size_t> reduce_axes; ExistentialQuantifierNode(AstNode* c, std::vector<size_t> ra) : child(c), reduce_axes(ra) {} };
struct UniversalQuantifierNode   : AstNode { AstNode* child; std::vector<size_t> reduce_axes; UniversalQuantifierNode(AstNode* c, std::vector<size_t> ra) : child(c), reduce_axes(ra) {} };
struct RelationalAggregateNode   : AstNode { AstNode* child; std::vector<size_t> retained_axes; RelationalAggregateNode(AstNode* c, std::vector<size_t> ra) : child(c), retained_axes(ra) {} };

// RISOLUTIVO: Parametrizzazione dell'Epoch Iteration per il GradientTape
struct EpochIterationNode : AstNode {
    uint64_t epochs;
    float base_learning_rate;
    int64_t optimizer_type; // 0=SGD, 1=Momentum, 2=Adam
    std::vector<AstNode*> gradient_parameters; // I parametri da ottimizzare (W, b)
    std::vector<AstNode*> computational_statements; // Le espressioni interne del loop

    EpochIterationNode(uint64_t ep, float lr, int64_t opt, std::vector<AstNode*> params, std::vector<AstNode*> body)
        : epochs(ep), base_learning_rate(lr), optimizer_type(opt), gradient_parameters(std::move(params)), computational_statements(std::move(body)) {}
};


#include <mach7/type_switchN-patterns.hpp> // Support for N-ary Match statement on patterns
#include <mach7/patterns/address.hpp>      // Address and dereference combinators
#include <mach7/patterns/bindings.hpp>     // Mach7 support for bindings on arbitrary UDT
#include <mach7/patterns/constructor.hpp>  // Support for constructor patterns
#include <mach7/patterns/equivalence.hpp>  // Equivalence combinator +
#include <mach7/patterns/primitive.hpp>    // Wildcard, variable and value patterns

namespace mch {
    template <> struct bindings<VariableNode>  { Members(VariableNode::name, VariableNode::metadata); };
    template <> struct bindings<LoadSafeNode>  { Members(LoadSafeNode::filepath, LoadSafeNode::layer); };
    template <> struct bindings<FromScalarNode> { Members(FromScalarNode::value, FromScalarNode::target_type); };

    template <> struct bindings<AddNode>       { Members(AddNode::left, AddNode::right); };
    template <> struct bindings<SubNode>       { Members(SubNode::left, SubNode::right); };
    template <> struct bindings<MatMulNode>    { Members(MatMulNode::left, MatMulNode::right); };
    template <> struct bindings<CrossNode>     { Members(CrossNode::left, CrossNode::right); };
    template <> struct bindings<HadamardNode>  { Members(HadamardNode::left, HadamardNode::right); };
    template <> struct bindings<ThetaJoinNode> { Members(ThetaJoinNode::left, ThetaJoinNode::right); };
    template <> struct bindings<ContractionNode> { Members(ContractionNode::left, ContractionNode::right, ContractionNode::l_axis, ContractionNode::r_axis); };

    template <> struct bindings<ApplyUnaryNode> { Members(ApplyUnaryNode::child, ApplyUnaryNode::op_name); };
    template <> struct bindings<ReduceSumNode>  { Members(ReduceSumNode::child, ReduceSumNode::axis); };
    template <> struct bindings<ReduceAllNode>  { Members(ReduceAllNode::child); };
    template <> struct bindings<CastingNode>    { Members(CastingNode::child, CastingNode::target_type); };
    template <> struct bindings<PrimitiveFloatExtractNode> { Members(PrimitiveFloatExtractNode::child); };
    template <> struct bindings<SliceTensorNode> { Members(SliceTensorNode::child, SliceTensorNode::axis, SliceTensorNode::start, SliceTensorNode::end); };
    template <> struct bindings<SqueezeAxesNode> { Members(SqueezeAxesNode::child, SqueezeAxesNode::axis); };
    template <> struct bindings<SegmentSumNode>  { Members(SegmentSumNode::child, SegmentSumNode::num_segments); };
    template <> struct bindings<OneHotNode>      { Members(OneHotNode::child, OneHotNode::depth); };
    template <> struct bindings<GatherNdNode>    { Members(GatherNdNode::source, GatherNdNode::indices); };
    template <> struct bindings<WhereNode>       { Members(WhereNode::child, WhereNode::off_value); };
    template <> struct bindings<BracketsNode>    { Members(BracketsNode::child, BracketsNode::coords); };

    template <> struct bindings<ExistentialQuantifierNode> { Members(ExistentialQuantifierNode::child, ExistentialQuantifierNode::reduce_axes); };
    template <> struct bindings<UniversalQuantifierNode>   { Members(UniversalQuantifierNode::child, UniversalQuantifierNode::reduce_axes); };
    template <> struct bindings<RelationalAggregateNode>   { Members(RelationalAggregateNode::child, RelationalAggregateNode::retained_axes); };

    // Binding del Nastro Parametrico delle Epoche
    template <> struct bindings<EpochIterationNode> {
        Members(EpochIterationNode::epochs, EpochIterationNode::base_learning_rate,
                 EpochIterationNode::gradient_parameters,
                EpochIterationNode::computational_statements);
    };
}


#endif //TENSORLIBRARY_JITASTNODES_H
