//
// Created by gyankos on 03/10/26.
//

#ifndef METATENSOR_ALLMACHS7_H
#define METATENSOR_ALLMACHS7_H

#include <mach7/type_switchN-patterns.hpp> // Support for N-ary Match statement on patterns
#include <mach7/patterns/address.hpp>      // Address and dereference combinators
#include <mach7/patterns/bindings.hpp>     // Mach7 support for bindings on arbitrary UDT
#include <mach7/patterns/constructor.hpp>  // Support for constructor patterns
#include <mach7/patterns/equivalence.hpp>  // Equivalence combinator +
#include <mach7/patterns/primitive.hpp>    // Wildcard, variable and value patterns

#include "LambdaAst.h"
#include "Metadata.h"

namespace mch {
    template <> struct bindings<CellTerminalNode>       {   };
    template <> struct bindings<FloatVariableIdNode>    { Members(FloatVariableIdNode::variable_id); };
    template <> struct bindings<FloatValueLiteralNode>  { Members(FloatValueLiteralNode::literal_value); };
    template <> struct bindings<IntrinsicPredicateNode> { Members(IntrinsicPredicateNode::predicate_type); };

    template <> struct bindings<FloatBinaryOpNode>      { Members(FloatBinaryOpNode::left, FloatBinaryOpNode::right, FloatBinaryOpNode::op); };
    template <> struct bindings<FloatUnaryOpNode>       { Members(FloatUnaryOpNode::child, FloatUnaryOpNode::op); };

    template <> struct bindings<FloatComparisonNode>    { Members(FloatComparisonNode::left, FloatComparisonNode::right, FloatComparisonNode::op); };
    template <> struct bindings<LogicalAndNode>         { Members(LogicalAndNode::left, LogicalAndNode::right); };
    template <> struct bindings<LogicalOrNode>          { Members(LogicalOrNode::left, LogicalOrNode::right); };
    template <> struct bindings<LogicalNotNode>         { Members(LogicalNotNode::child); };
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
    template<> struct bindings<PermuteAxesNode> {
        Members(PermuteAxesNode::child, PermuteAxesNode::axes);
    };
    template <> struct bindings<ProgramRootNode> { Members(ProgramRootNode::configurations, ProgramRootNode::execution_graph); };
    template <> struct bindings<DumpSafeNode> { Members(DumpSafeNode::tensor_node, DumpSafeNode::filepath, DumpSafeNode::tensor_name); };
    template <> struct bindings<ExportStatementNode> { Members(ExportStatementNode::output_nodes); };
};


#endif //METATENSOR_ALLMACHS7_H
