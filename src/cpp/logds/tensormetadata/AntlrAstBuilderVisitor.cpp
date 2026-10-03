//
// Created by gyankos on 03/10/26.
//
#include "logds/tensormetadata/AntlrAstBuilderVisitor.h"

std::any AntlrAstBuilderVisitor::visitTensorDeclaration(TensorLangParser::TensorDeclarationContext *context) {
    std::string var_name = context->ID()->getText();
    auto metadata = parse_type_id(context->typeID());

    TypingNodePtr raw_var = std::make_shared<VariableNode>(var_name, metadata);
    symbol_table[var_name] = raw_var;

    // Decoriamo immediatamente l'astrazione di tipo all'interno del Flyweight
    auto flyweight_node = std::make_shared<JitFlyweightDecorator>(raw_var);

    // Se a destra c'è un'espressione complessa (es: load), ne tracciamo i percorsi estrinseci
    if (auto* load_ctx = dynamic_cast<TensorLangParser::LoadExprContext*>(context->expr())) {
        flyweight_node->set_extrinsic_filepath(clean_string(load_ctx->STRING(0)->getText()));
    }

    return std::any(std::static_pointer_cast<TensorTyping>(flyweight_node));
}
