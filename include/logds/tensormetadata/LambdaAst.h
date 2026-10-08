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

#ifndef METATENSOR_LAMBDAAST_H
#define METATENSOR_LAMBDAAST_H

#include <string>
#include <vector>
#include <memory>


// =============================================================================
// INTERFACCE COREGRAFICHE PARALLELE DELL'AST PER LE LAMBDA
// =============================================================================

// Nodo Radice generico per le espressioni numeriche e logiche delle Lambda
struct LambdaExpressionNode {
    virtual ~LambdaExpressionNode() = default;
};

using LambdaAstPtr = std::shared_ptr<const LambdaExpressionNode>;

// Sotto-interfaccia specifica per i nodi che restituiscono valori fluttuanti (Float)
struct FloatExpressionNode : LambdaExpressionNode {};
using FloatExprPtr = std::shared_ptr<const FloatExpressionNode>;

// Sotto-interfaccia specifica per i nodi che restituiscono condizioni booleane (Comparazioni)
struct BooleanConditionNode : LambdaExpressionNode {};
using BoolCondPtr = std::shared_ptr<const BooleanConditionNode>;

// =============================================================================
// I TRE NODI BASE DI TIPO FLOAT (FOGLIE DELL'ALBERO)
// =============================================================================

// 1. La parola chiave 'cell': rappresenta il valore float corrente analizzato a runtime
struct CellTerminalNode : FloatExpressionNode {};

// 2. Un identificatore di variabile float interna al programma (es: "bias", "threshold")
struct FloatVariableIdNode : FloatExpressionNode {
    std::string variable_id;
    FloatVariableIdNode(std::string id) : variable_id(std::move(id)) {}
};

// 3. Un valore letterale fluttuante costante (es: 0.0f, -1.0f, 1e-5f)
struct FloatValueLiteralNode : FloatExpressionNode {
    float literal_value;
    FloatValueLiteralNode(float val) : literal_value(val) {}
};

// =============================================================================
// ENUMERAZIONI OPERAZIONALI E COSTANTI INTRESECCHE IMPLICITE
// =============================================================================
enum class FloatBinaryOp        { ADD, SUB, DIV, MUL }; // +, -, /, *
enum class FloatUnaryOp         { ABS };           // abs
enum class FloatCompOp          { LT, GT, LE, GE, EQ, NE }; // <, >, <=, >=, ==, !=
enum class IntrinsicPred        { ISNAN, EPS, PLUS_INFINITY, MINUS_INFINITY };
enum class LogicalBinOp         { AND, OR, XOR };
enum class LogicalUnOp          { NOT };
enum class TensorAggregation    { ALL, ANY, ALL_SUM };
enum class BinaryTensorOp       { MM_2DMatrixMult, MatMul, Cross, ADD, ElementWise_MUL };

// Costanti intrinseche implicite agenti sulla cella
struct IntrinsicPredicateNode : BooleanConditionNode {
    IntrinsicPred predicate_type;
    IntrinsicPredicateNode(IntrinsicPred pred) : predicate_type(pred) {}
};

// =============================================================================
// PROIEZIONI ARITMETICHE SCALARI (Float expressions over base nodes)
// =============================================================================
struct FloatBinaryOpNode : FloatExpressionNode {
    FloatExprPtr left;
    FloatExprPtr right;
    FloatBinaryOp op;

    FloatBinaryOpNode(FloatExprPtr l, FloatExprPtr r, FloatBinaryOp o)
        : left(std::move(l)), right(std::move(r)), op(o) {}
};

struct FloatUnaryOpNode : FloatExpressionNode {
    FloatExprPtr child;
    FloatUnaryOp op;

    FloatUnaryOpNode(FloatExprPtr c, FloatUnaryOp o)
        : child(std::move(c)), op(o) {}
};

// =============================================================================
// COMPARAZIONI ED ESPRESSIONI LOGICHE BOOLEANE
// =============================================================================

// Confronta due espressioni float arbitrarie (celle, variabili o letterali) via COMP_OP
struct FloatComparisonNode : BooleanConditionNode {
    FloatExprPtr left;
    FloatExprPtr right;
    FloatCompOp op;

    FloatComparisonNode(FloatExprPtr l, FloatExprPtr r, FloatCompOp o)
        : left(std::move(l)), right(std::move(r)), op(o) {}
};

// Connettivi logici di aggregazione per i predicati (and, or, not)
struct LogicalAndNode : BooleanConditionNode {
    BoolCondPtr left;
    BoolCondPtr right;
    LogicalAndNode(BoolCondPtr l, BoolCondPtr r) : left(std::move(l)), right(std::move(r)) {}
};

struct LogicalOrNode : BooleanConditionNode {
    BoolCondPtr left;
    BoolCondPtr right;
    LogicalOrNode(BoolCondPtr l, BoolCondPtr r) : left(std::move(l)), right(std::move(r)) {}
};

struct LogicalNotNode : BooleanConditionNode {
    BoolCondPtr child;
    LogicalNotNode(BoolCondPtr c) : child(std::move(c)) {}
};





#endif //METATENSOR_LAMBDAAST_H
