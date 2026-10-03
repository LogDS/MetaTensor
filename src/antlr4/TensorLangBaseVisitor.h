
// Generated from src/antlr4/TensorLang.g4 by ANTLR 4.13.2

#pragma once


#include "antlr4-runtime.h"
#include "TensorLangVisitor.h"


/**
 * This class provides an empty implementation of TensorLangVisitor, which can be
 * extended to create a visitor which only needs to handle a subset of the available methods.
 */
class  TensorLangBaseVisitor : public TensorLangVisitor {
public:

  virtual std::any visitProgram(TensorLangParser::ProgramContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitDeviceConfig(TensorLangParser::DeviceConfigContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitClusterConfig(TensorLangParser::ClusterConfigContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitOptimizerConfig(TensorLangParser::OptimizerConfigContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitDecayConfig(TensorLangParser::DecayConfigContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitStatement(TensorLangParser::StatementContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitTensorDeclaration(TensorLangParser::TensorDeclarationContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitTypeID(TensorLangParser::TypeIDContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitDatatype(TensorLangParser::DatatypeContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitLayout(TensorLangParser::LayoutContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitDimSeq(TensorLangParser::DimSeqContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitDimToken(TensorLangParser::DimTokenContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitActiveEpochBlock(TensorLangParser::ActiveEpochBlockContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitAssignment(TensorLangParser::AssignmentContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitCrossExpr(TensorLangParser::CrossExprContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitThetaJoinExpr(TensorLangParser::ThetaJoinExprContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitExistentialExpr(TensorLangParser::ExistentialExprContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitIdExpr(TensorLangParser::IdExprContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitScalarExpr(TensorLangParser::ScalarExprContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitReduceAllExpr(TensorLangParser::ReduceAllExprContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitSubExpr(TensorLangParser::SubExprContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitUniversalExpr(TensorLangParser::UniversalExprContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitAddExpr(TensorLangParser::AddExprContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitUnaryExpr(TensorLangParser::UnaryExprContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitPermuteExpr(TensorLangParser::PermuteExprContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitGatherExpr(TensorLangParser::GatherExprContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitLoadExpr(TensorLangParser::LoadExprContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitHadamardExpr(TensorLangParser::HadamardExprContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitReduceAxisExpr(TensorLangParser::ReduceAxisExprContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitMatMulExpr(TensorLangParser::MatMulExprContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitLambdaPred(TensorLangParser::LambdaPredContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitLogicalExpr(TensorLangParser::LogicalExprContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitLogicalAndExpr(TensorLangParser::LogicalAndExprContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitNotPredicate(TensorLangParser::NotPredicateContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitSubPredicate(TensorLangParser::SubPredicateContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitCompPredicate(TensorLangParser::CompPredicateContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitIntrinsicPredicate(TensorLangParser::IntrinsicPredicateContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitFloatComparison(TensorLangParser::FloatComparisonContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitFloatExpr(TensorLangParser::FloatExprContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitFloatTerm(TensorLangParser::FloatTermContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitAbsFloatExpr(TensorLangParser::AbsFloatExprContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitCellFloatExpr(TensorLangParser::CellFloatExprContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitIdFloatExpr(TensorLangParser::IdFloatExprContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitLiteralFloatExpr(TensorLangParser::LiteralFloatExprContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitNegateFloatExpr(TensorLangParser::NegateFloatExprContext *ctx) override {
    return visitChildren(ctx);
  }


};

