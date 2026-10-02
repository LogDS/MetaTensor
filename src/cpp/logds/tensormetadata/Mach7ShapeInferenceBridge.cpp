//
// Created by gyankos on 02/10/26.
//

#include <logds/tensormetadata/Mach7ShapeInferenceBridge.h>
#include <mach7/type_switchN-patterns.hpp> // Support for N-ary Match statement on patterns
#include <mach7/patterns/address.hpp>      // Address and dereference combinators
#include <mach7/patterns/bindings.hpp>     // Mach7 support for bindings on arbitrary UDT
#include <mach7/patterns/constructor.hpp>  // Support for constructor patterns
#include <mach7/patterns/equivalence.hpp>  // Equivalence combinator +
#include <mach7/patterns/primitive.hpp>    // Wildcard, variable and value patterns
using namespace mch;

std::optional<JitTensorMetadata> Mach7ShapeInferenceBridge::infer_node_shape(AstNode *node, const JitShapeInferenceEngine &engine) {
    var<std::string> s; var<DataType> dt; var<uint64_t> u1, u2; var<size_t> ax1, ax2;
    var<float> f1; var<int64_t> opt;
    var< AstNode*> c_node, r_node;
    var<std::vector<size_t>> v_axes;
    var<std::vector<uint64_t>> v_coords;
    var<std::vector<AstNode*>> v_params, v_body;
    var<JitTensorMetadata> meta_data;

    Match(node) {
            Case(C<VariableNode>(s, meta_data))     return meta_data;
            Case(C<FromScalarNode>(f1, dt))         return JitTensorMetadata{dt, StorageLayout::Dense, 0, {}, true};

            Case(C<AddNode>(c_node, r_node)) {
                auto L = infer_node_shape(c_node, engine); auto R = infer_node_shape(r_node, engine);
                if (L && R) return engine.operator_add(*L, *R); // Invocazione di istanza contestuale
                return std::nullopt;
            }
            Case(C<SubNode>(c_node, r_node)) {
                auto L = infer_node_shape(c_node, engine); auto R = infer_node_shape(r_node, engine);
                if (L && R) return engine.operator_sub(*L, *R);
                return std::nullopt;
            }
            Case(C<MatMulNode>(c_node, r_node)) {
                auto L = infer_node_shape(c_node, engine); auto R = infer_node_shape(r_node, engine);
                if (L && R) return engine.operator_matmul(*L, *R);
                return std::nullopt;
            }
            Case(C<CrossNode>(c_node, r_node)) {
                auto L = infer_node_shape(c_node, engine); auto R = infer_node_shape(r_node, engine);
                if (L && R) return engine.operator_cross(*L, *R);
                return std::nullopt;
            }
            Case(C<HadamardNode>(c_node, r_node)) {
                auto L = infer_node_shape(c_node, engine); auto R = infer_node_shape(r_node, engine);
                if (L && R) return engine.element_wise_mul(*L, *R);
                return std::nullopt;
            }
            Case(C<ThetaJoinNode>(c_node, r_node)) {
                auto L = infer_node_shape(c_node, engine); auto R = infer_node_shape(r_node, engine);
                if (L && R) return engine.tensor_theta_join(*L, *R);
                return std::nullopt;
            }
            Case(C<ContractionNode>(c_node, r_node, ax1, ax2)) {
                auto L = infer_node_shape(c_node, engine); auto R = infer_node_shape(r_node, engine);
                if (L && R) return engine.contraction(*L, *R, ax1, ax2);
                return std::nullopt;
            }

            Case(C<ApplyUnaryNode>(c_node, s)) {
                auto In = infer_node_shape(c_node, engine);
                if (In) return engine.apply(*In, s);
                return std::nullopt;
            }
            Case(C<ReduceSumNode>(c_node, ax1)) {
                auto In = infer_node_shape(c_node, engine);
                if (In) return engine.reduce_sum(*In, ax1);
                return std::nullopt;
            }
            Case(C<ReduceAllNode>(c_node)) {
                auto In = infer_node_shape(c_node, engine);
                if (In) return engine.reduce_all_sum(*In);
                return std::nullopt;
            }
            Case(C<CastingNode>(c_node, dt)) {
                auto In = infer_node_shape(c_node, engine);
                if (In) return engine.casting(*In, dt);
                return std::nullopt;
            }
            Case(C<PrimitiveFloatExtractNode>(c_node)) {
                auto In = infer_node_shape(c_node, engine);
                if (In) return engine.float_extraction(*In);
                return std::nullopt;
            }
            Case(C<SliceTensorNode>(c_node, ax1, u1, u2)) {
                auto In = infer_node_shape(c_node, engine);
                if (In) return engine.slice_tensor(*In, ax1, u1, u2);
                return std::nullopt;
            }
            Case(C<SqueezeAxesNode>(c_node, ax1)) {
                auto In = infer_node_shape(c_node, engine);
                if (In) return engine.squeeze_axes(*In, ax1);
                return std::nullopt;
            }
            Case(C<SegmentSumNode>(c_node, u1)) {
                auto In = infer_node_shape(c_node, engine);
                if (In) return engine.segment_sum(*In, u1);
                return std::nullopt;
            }
            Case(C<OneHotNode>(c_node, u1)) {
                auto In = infer_node_shape(c_node, engine);
                if (In) return engine.one_hot_encoding(*In, u1);
                return std::nullopt;
            }
            Case(C<GatherNdNode>(c_node, r_node)) {
                auto S = infer_node_shape(c_node, engine); auto I = infer_node_shape(r_node, engine);
                if (S && I) return engine.gather_nd(*S, *I);
                return std::nullopt;
            }
            Case(C<WhereNode>(c_node, f1)) {
                auto In = infer_node_shape(c_node, engine);
                if (In) return engine.where(*In, f1);
                return std::nullopt;
            }
            Case(C<BracketsNode>(c_node, v_coords)) {
                auto In = infer_node_shape(c_node, engine);
                if (In) return engine.operator_brackets(*In, v_coords);
                return std::nullopt;
            }

            Case(C<ExistentialQuantifierNode>(c_node, v_axes)) {
                auto In = infer_node_shape(c_node, engine);
                if (In) return engine.evaluate_existential(*In, v_axes);
                return std::nullopt;
            }
            Case(C<UniversalQuantifierNode>(c_node, v_axes)) {
                auto In = infer_node_shape(c_node, engine);
                if (In) return engine.evaluate_universal(*In, v_axes);
                return std::nullopt;
            }
            Case(C<RelationalAggregateNode>(c_node, v_axes)) {
                auto In = infer_node_shape(c_node, engine);
                if (In) return engine.aggregate(*In, v_axes);
                return std::nullopt;
            }

            Case(C<EpochIterationNode>(u1, f1, v_params, v_body)) {
                std::optional<JitTensorMetadata> final_loss_meta;
                for (auto* statement_node : v_body) {
                    final_loss_meta = infer_node_shape(statement_node, engine);
                }
                if (!final_loss_meta || final_loss_meta->rank != 0) return std::nullopt;

                for (auto* param_node : v_params) {
                    auto param_meta = infer_node_shape(param_node, engine);
                    // I gradienti ereditano la forma contestuale normalizzata
                }
                return final_loss_meta;
            }

            Otherwise() { return std::nullopt; }
            }
    EndMatch
}
