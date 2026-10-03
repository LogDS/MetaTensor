
// Generated from src/antlr4/TensorLang.g4 by ANTLR 4.13.2

#pragma once


#include "antlr4-runtime.h"
#include "TensorLangParser.h"



/**
 * This class defines an abstract visitor for a parse tree
 * produced by TensorLangParser.
 */
class  TensorLangVisitor : public antlr4::tree::AbstractParseTreeVisitor {
public:

  /**
   * Visit parse trees produced by TensorLangParser.
   */
    virtual std::any visitProgram(TensorLangParser::ProgramContext *context) = 0;

    virtual std::any visitDeviceConfig(TensorLangParser::DeviceConfigContext *context) = 0;

    virtual std::any visitClusterConfig(TensorLangParser::ClusterConfigContext *context) = 0;

    virtual std::any visitOptimizerConfig(TensorLangParser::OptimizerConfigContext *context) = 0;

    virtual std::any visitDecayConfig(TensorLangParser::DecayConfigContext *context) = 0;

    virtual std::any visitStatement(TensorLangParser::StatementContext *context) = 0;

    virtual std::any visitTensorDeclaration(TensorLangParser::TensorDeclarationContext *context) = 0;

    virtual std::any visitTypeID(TensorLangParser::TypeIDContext *context) = 0;

    virtual std::any visitDatatype(TensorLangParser::DatatypeContext *context) = 0;

    virtual std::any visitLayout(TensorLangParser::LayoutContext *context) = 0;

    virtual std::any visitDimSeq(TensorLangParser::DimSeqContext *context) = 0;

    virtual std::any visitDimToken(TensorLangParser::DimTokenContext *context) = 0;

    virtual std::any visitActiveEpochBlock(TensorLangParser::ActiveEpochBlockContext *context) = 0;

    virtual std::any visitAssignment(TensorLangParser::AssignmentContext *context) = 0;

    virtual std::any visitCrossExpr(TensorLangParser::CrossExprContext *context) = 0;

    virtual std::any visitThetaJoinExpr(TensorLangParser::ThetaJoinExprContext *context) = 0;

    virtual std::any visitExistentialExpr(TensorLangParser::ExistentialExprContext *context) = 0;

    virtual std::any visitIdExpr(TensorLangParser::IdExprContext *context) = 0;

    virtual std::any visitScalarExpr(TensorLangParser::ScalarExprContext *context) = 0;

    virtual std::any visitReduceAllExpr(TensorLangParser::ReduceAllExprContext *context) = 0;

    virtual std::any visitSubExpr(TensorLangParser::SubExprContext *context) = 0;

    virtual std::any visitUniversalExpr(TensorLangParser::UniversalExprContext *context) = 0;

    virtual std::any visitAddExpr(TensorLangParser::AddExprContext *context) = 0;

    virtual std::any visitUnaryExpr(TensorLangParser::UnaryExprContext *context) = 0;

    virtual std::any visitPermuteExpr(TensorLangParser::PermuteExprContext *context) = 0;

    virtual std::any visitGatherExpr(TensorLangParser::GatherExprContext *context) = 0;

    virtual std::any visitLoadExpr(TensorLangParser::LoadExprContext *context) = 0;

    virtual std::any visitHadamardExpr(TensorLangParser::HadamardExprContext *context) = 0;

    virtual std::any visitReduceAxisExpr(TensorLangParser::ReduceAxisExprContext *context) = 0;

    virtual std::any visitMatMulExpr(TensorLangParser::MatMulExprContext *context) = 0;

    virtual std::any visitLambdaPred(TensorLangParser::LambdaPredContext *context) = 0;

    virtual std::any visitLogicalExpr(TensorLangParser::LogicalExprContext *context) = 0;

    virtual std::any visitLogicalAndExpr(TensorLangParser::LogicalAndExprContext *context) = 0;

    virtual std::any visitNotPredicate(TensorLangParser::NotPredicateContext *context) = 0;

    virtual std::any visitSubPredicate(TensorLangParser::SubPredicateContext *context) = 0;

    virtual std::any visitCompPredicate(TensorLangParser::CompPredicateContext *context) = 0;

    virtual std::any visitIntrinsicPredicate(TensorLangParser::IntrinsicPredicateContext *context) = 0;

    virtual std::any visitFloatComparison(TensorLangParser::FloatComparisonContext *context) = 0;

    virtual std::any visitFloatExpr(TensorLangParser::FloatExprContext *context) = 0;

    virtual std::any visitFloatTerm(TensorLangParser::FloatTermContext *context) = 0;

    virtual std::any visitAbsFloatExpr(TensorLangParser::AbsFloatExprContext *context) = 0;

    virtual std::any visitCellFloatExpr(TensorLangParser::CellFloatExprContext *context) = 0;

    virtual std::any visitIdFloatExpr(TensorLangParser::IdFloatExprContext *context) = 0;

    virtual std::any visitLiteralFloatExpr(TensorLangParser::LiteralFloatExprContext *context) = 0;

    virtual std::any visitNegateFloatExpr(TensorLangParser::NegateFloatExprContext *context) = 0;


};

