//
// Created by gyankos on 03/10/26.
//

#ifndef METATENSOR_ANTLRASTBUILDERVISITOR_H
#define METATENSOR_ANTLRASTBUILDERVISITOR_H


#include <TensorLangBaseVisitor.h>

#include "logds/tensormetadata/JitFlyweightDecorator.h"
#include "logds/tensormetadata/LambdaAst.h"
#include <unordered_map>
#include <string>
#include <vector>
#include <any>
#include <memory>
#include <stdexcept>
#include <algorithm>

class AntlrAstBuilderVisitor : public TensorLangBaseVisitor {
private:
    std::unordered_map<std::string, TypingNodePtr> symbol_table;
    std::vector<TypingNodePtr> active_configurations;

    static std::string clean_string(const std::string& str) {
        if (str.length() >= 2 && str.front() == '"' && str.back() == '"') {
            return str.substr(1, str.length() - 2);
        }
        return str;
    }

    static std::vector<std::variant<std::string, uint64_t>> parse_dim_seq(TensorLangParser::DimSeqContext* ctx) {
        std::vector<std::variant<std::string, uint64_t>> shape;
        for (auto* token_node : ctx->dimToken()) {
            std::string s = token_node->getText();
            if (token_node->INT()) shape.push_back(static_cast<uint64_t>(std::stoull(s)));
            else shape.push_back(s);
        }
        return shape;
    }

    static JitTensorMetadata parse_type_id(TensorLangParser::TypeIDContext* ctx) {
        DataType dt = TYPE_FLOAT;
        std::string dt_str = ctx->datatype()->getText();
        if (dt_str == "double") dt = TYPE_DOUBLE;
        else if (dt_str == "int") dt = TYPE_INT32;

        StorageLayout layout = (ctx->layout()->getText() == "SparseCOO") ? StorageLayout::SparseCOO : StorageLayout::Dense;
        auto shape = parse_dim_seq(ctx->dimSeq());
        return JitTensorMetadata{dt, layout, shape.size(), shape, true};
    }

public:
    AntlrAstBuilderVisitor() = default;

    // =============================================================================
    // I. DICHIARAZIONI DI TIPO E CONFIGURAZIONI (TensorTyping + Flyweight)
    // =============================================================================
    std::any visitTensorDeclaration(TensorLangParser::TensorDeclarationContext *context) override;

    std::any visitActiveEpochBlock(TensorLangParser::ActiveEpochBlockContext *context) override {
        uint64_t epochs = std::stoull(context->INT()->getText());
        std::string weight_name = context->ID()->getText();
        float lr = std::stof(context->FLOAT()->getText());

        std::vector<TypingNodePtr> loop_body;
        for (auto* stmt_ctx : context->statement()) {
            std::any res = stmt_ctx->accept(this);
            if (res.has_value()) {
                auto dec = std::any_cast<TypingNodePtr>(res);
                loop_body.push_back(std::static_pointer_cast<JitFlyweightDecorator>(dec)->get_intrinsic_type());
            }
        }

        std::vector<TypingNodePtr> params = { symbol_table[weight_name] };
        TypingNodePtr epoch_raw = std::make_shared<EpochIterationNode>(epochs, lr, 0, params, loop_body);

        return std::any(std::make_shared<JitFlyweightDecorator>(epoch_raw));
    }

    // =============================================================================
    // II. OPERATORI GRAFO COMPUTACOLO TENSORIALE (Downstream Decorator Bridges)
    // =============================================================================
    std::any visitMatMulExpr(TensorLangParser::MatMulExprContext *context) override {
        auto L_dec = std::any_cast<TypingNodePtr>(visit(context->expr(0)));
        auto R_dec = std::any_cast<TypingNodePtr>(visit(context->expr(1)));

        auto L_raw = std::static_pointer_cast<JitFlyweightDecorator>(L_dec)->get_intrinsic_type();
        auto R_raw = std::static_pointer_cast<JitFlyweightDecorator>(R_dec)->get_intrinsic_type();

        return std::any(std::make_shared<JitFlyweightDecorator>(std::make_shared<MatMulNode>(L_raw, R_raw)));
    }

    std::any visitExistentialExpr(TensorLangParser::ExistentialExprContext *context) override {
        auto child_dec = std::any_cast<TypingNodePtr>(context->expr()->accept(this));
        auto child_raw = std::static_pointer_cast<JitFlyweightDecorator>(child_dec)->get_intrinsic_type();
        auto axes = parse_dim_seq(context->dimSeq());

        // Forgiamo il nodo intrinseco di tipo puro
        TypingNodePtr exist_raw = std::make_shared<ExistentialQuantifierNode>(child_raw, v_axes_convert(axes));
        auto flyweight_term = std::make_shared<JitFlyweightDecorator>(exist_raw);

        // VISITA AST COMPILATORE DELLE LAMBDA PARALLELO: Popola l'estrinseco a costo zero
        auto lambda_ast_root = std::any_cast<LambdaAstPtr>(context->lambdaPred()->accept(this));
        flyweight_term->set_extrinsic_lambda_graph(lambda_ast_root);

        return std::any(std::static_pointer_cast<TensorTyping>(flyweight_term));
    }

    // =============================================================================
    // III. IL NUOVO COMPILATORE AST DELLE ESPRESSIONI LAMBDA PARALLELO (`LambdaAst`)
    // =============================================================================
    std::any visitLambdaPred(TensorLangParser::LambdaPredContext *context) override {
        // La visita della catena logica restituisce la radice BoolCondPtr dell'AST parallelo
        return context->logicalExpr()->accept(this);
    }

    std::any visitLogicalExpr(TensorLangParser::LogicalExprContext *context) override {
        auto current_node = std::any_cast<BoolCondPtr>(context->logicalAndExpr(0)->accept(this));
        for (size_t i = 1; i < context->logicalAndExpr().size(); ++i) {
            auto next_node = std::any_cast<BoolCondPtr>(context->logicalAndExpr(i)->accept(this));
            current_node = std::make_shared<const LogicalOrNode>(current_node, next_node);
        }
        return std::any(std::static_pointer_cast<const LambdaExpressionNode>(current_node));
    }

    std::any visitIntrinsicPredicate(TensorLangParser::IntrinsicPredicateContext *context) override {
        std::string pred_str = context->INTRINSIC_OP()->getText();
        IntrinsicPred token = IntrinsicPred::ISNAN;
        if (pred_str == "eps") token = IntrinsicPred::EPS;
        else if (pred_str == "plus_infinity") token = IntrinsicPred::PLUS_INFINITY;
        else if (pred_str == "minus_infinity") token = IntrinsicPred::MINUS_INFINITY;

        BoolCondPtr node = std::make_shared<const IntrinsicPredicateNode>(token);
        return std::any(std::static_pointer_cast<const LambdaExpressionNode>(node));
    }

    std::any visitCompPredicate(TensorLangParser::CompPredicateContext *context) override {
        auto L = std::any_cast<FloatExprPtr>(context->floatComparison()->floatExpr(0)->accept(this));
        auto R = std::any_cast<FloatExprPtr>(context->floatComparison()->floatExpr(1)->accept(this));

        std::string op_str = context->floatComparison()->COMP_OP()->getText();
        FloatCompOp op = FloatCompOp::EQ;
        if (op_str == ">") op = FloatCompOp::GT;
        else if (op_str == "<") op = FloatCompOp::LT;
        else if (op_str == ">=") op = FloatCompOp::GE;
        else if (op_str == "<=") op = FloatCompOp::LE;
        else if (op_str == "!=") op = FloatCompOp::NE;

        BoolCondPtr node = std::make_shared<const FloatComparisonNode>(L, R, op);
        return std::any(std::static_pointer_cast<const LambdaExpressionNode>(node));
    }

    std::any visitCellFloatExpr(TensorLangParser::CellFloatExprContext*) override {
        FloatExprPtr cell = std::make_shared<const CellTerminalNode>();
        return std::any(std::static_pointer_cast<const LambdaExpressionNode>(cell));
    }

    std::any visitIdFloatExpr(TensorLangParser::IdFloatExprContext *context) override {
        FloatExprPtr var_node = std::make_shared<const FloatVariableIdNode>(context->ID()->getText());
        return std::any(std::static_pointer_cast<const LambdaExpressionNode>(var_node));
    }

    std::any visitLiteralFloatExpr(TensorLangParser::LiteralFloatExprContext *context) override {
        float val = std::stof(context->FLOAT()->getText());
        FloatExprPtr lit = std::make_shared<const FloatValueLiteralNode>(val);
        return std::any(std::static_pointer_cast<const LambdaExpressionNode>(lit));
    }

    // --- PUNTO DI INGRESSO (Program Radice) ---
    std::any visitProgram(TensorLangParser::ProgramContext *context) override {
        for (auto* cfg : context->configDirective()) cfg->accept(this);
        std::vector<TypingNodePtr> bodies;
        for (auto* stmt : context->statement()) {
            std::any res = visit(stmt);
            if (res.has_value()) {
                auto dec = std::any_cast<TypingNodePtr>(res);
                bodies.push_back(std::static_pointer_cast<JitFlyweightDecorator>(dec)->get_intrinsic_type());
            }
        }
        TypingNodePtr root_raw = std::make_shared<ProgramRootNode>(active_configurations, bodies);
        return std::any(root_raw);
    }
private:
    // Helper privato per convertire le varianti nei vettori numerici grezzi degli assi attesi dall'AST originale
    std::vector<size_t> v_axes_convert(const std::vector<std::variant<std::string, uint64_t>>& src) {
        std::vector<size_t> dest;
        for (const auto& val : src) {
            if (std::holds_alternative<uint64_t>(val)) dest.push_back(std::get<uint64_t>(val));
            else dest.push_back(0);
        }
        return dest;
    }
};
#endif //METATENSOR_ANTLRASTBUILDERVISITOR_H
