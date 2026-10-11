//
// Created by Giacomo Bergami, PhD on 11/10/26.
//

#ifndef METATENSOR_METATENSORLANGUAGE_H
#define METATENSOR_METATENSORLANGUAGE_H
#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>
#include <rva/variant.hpp>
#include <array>

enum class LambdaArgOp {
    LAO_LT,
    LAO_LEQ,
    LAO_GT,
    LAO_GEQ,
    LAO_EQ,
    LAO_NEQ,
    LAO_LAND,
    LAO_LOR,
    LAO_LXOR,
    LAO_LNEG,
};

struct LambdaArg {
    LambdaArgOp op;
    std::vector<std::variant<std::string, double, LambdaArg>> args;
};

enum class BoolTensorOp {
    LTO_INDICATOR,
    LTO_FORALL,
    LTO_EXISTS
};

struct BoolTensor {
    BoolTensorOp op;
    std::variant<std::shared_ptr<BoolTensor>, LambdaArg> object;
};

using Boolean = std::variant<std::string, bool, std::pair<BoolTensorOp, BoolTensor>>;

enum class FloatOp {
    FO_PLUS,
    FO_MINUS,
    FO_TIMES,
    FO_DIVIDE,
};

enum class BinaryTensorOp {
    BTO_PROD,
    BTO_HADAMARD,
    BTO_SUM,
    BOT_DIFF,
    BOT_GATHER,
    BOT_RESHAPE
};

enum class BinaryTensorFloatOp {
    BTFO_PLUS,
    BTFO_TIMES,
    BTFO_DIV,
    BTFO_MINUS,
};

enum class TensorReduceOP {
    TROP_MAX,
    TROP_MIN,
    TROP_SUM,
    TROP_PROD
};

enum class PartialAggrTensor {
    PERMUTATION,
    SUM,
    PRODUCT,
};

enum class SpuriousTensorOp {
    TENSOR_FLOAT_CLIP,
    BOLTENSOR_TENSOR_TENSOR_CLIP
};

enum class NAryTensorOp {
    NTO_STACK
};
struct TensorRef;

using Float = rva::variant<double,
                           std::string,
                           std::pair<TensorReduceOP, std::vector<TensorRef>>,
                           std::pair<BinaryTensorFloatOp, std::vector<rva::self_t>>
                           >;

struct TensorRef {
    std::variant<BinaryTensorOp, BinaryTensorFloatOp, PartialAggrTensor, SpuriousTensorOp, NAryTensorOp> op;
    std::string var_id;
    std::vector<TensorRef> lhs_rhs;
    std::vector<Float> rhs_float;
    std::vector<uint64_t> tensor_indices;
    std::optional<BoolTensor> for_clipping;

    bool isVarId() const {
        return !var_id.empty();
    }
};

using Term = std::variant<BoolTensor, Boolean, Float, TensorRef>;

enum class TLTypes {
    T_BOOL,
    T_FLOAT32,
    T_FLOAT64,
    T_INT32,
    T_INT64,
    T_UINT32
};

using DimensionForTypes = std::variant<std::string, uint64_t>;
using AnyType = std::variant<TLTypes,
                              std::pair<std::vector<DimensionForTypes>, TLTypes>>;

enum class StatementType {
    S_WHILE,
    S_IFTE,
    S_LETIN
};

struct Statement {
    Boolean condition;
    std::vector<std::vector<Statement>> block;
    std::optional<std::pair<Term, AnyType>> declaration;
};

#endif //METATENSOR_METATENSORLANGUAGE_H
