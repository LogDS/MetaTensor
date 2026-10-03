
// Generated from src/antlr4/TensorLang.g4 by ANTLR 4.13.2

#pragma once


#include "antlr4-runtime.h"
#include "TensorLangParser.h"


/**
 * This interface defines an abstract listener for a parse tree produced by TensorLangParser.
 */
class  TensorLangListener : public antlr4::tree::ParseTreeListener {
public:

  virtual void enterProgram(TensorLangParser::ProgramContext *ctx) = 0;
  virtual void exitProgram(TensorLangParser::ProgramContext *ctx) = 0;

  virtual void enterDeviceConfig(TensorLangParser::DeviceConfigContext *ctx) = 0;
  virtual void exitDeviceConfig(TensorLangParser::DeviceConfigContext *ctx) = 0;

  virtual void enterClusterConfig(TensorLangParser::ClusterConfigContext *ctx) = 0;
  virtual void exitClusterConfig(TensorLangParser::ClusterConfigContext *ctx) = 0;

  virtual void enterOptimizerConfig(TensorLangParser::OptimizerConfigContext *ctx) = 0;
  virtual void exitOptimizerConfig(TensorLangParser::OptimizerConfigContext *ctx) = 0;

  virtual void enterDecayConfig(TensorLangParser::DecayConfigContext *ctx) = 0;
  virtual void exitDecayConfig(TensorLangParser::DecayConfigContext *ctx) = 0;

  virtual void enterStatement(TensorLangParser::StatementContext *ctx) = 0;
  virtual void exitStatement(TensorLangParser::StatementContext *ctx) = 0;

  virtual void enterTensorDeclaration(TensorLangParser::TensorDeclarationContext *ctx) = 0;
  virtual void exitTensorDeclaration(TensorLangParser::TensorDeclarationContext *ctx) = 0;

  virtual void enterTypeID(TensorLangParser::TypeIDContext *ctx) = 0;
  virtual void exitTypeID(TensorLangParser::TypeIDContext *ctx) = 0;

  virtual void enterDatatype(TensorLangParser::DatatypeContext *ctx) = 0;
  virtual void exitDatatype(TensorLangParser::DatatypeContext *ctx) = 0;

  virtual void enterLayout(TensorLangParser::LayoutContext *ctx) = 0;
  virtual void exitLayout(TensorLangParser::LayoutContext *ctx) = 0;

  virtual void enterDimSeq(TensorLangParser::DimSeqContext *ctx) = 0;
  virtual void exitDimSeq(TensorLangParser::DimSeqContext *ctx) = 0;

  virtual void enterDimToken(TensorLangParser::DimTokenContext *ctx) = 0;
  virtual void exitDimToken(TensorLangParser::DimTokenContext *ctx) = 0;

  virtual void enterActiveEpochBlock(TensorLangParser::ActiveEpochBlockContext *ctx) = 0;
  virtual void exitActiveEpochBlock(TensorLangParser::ActiveEpochBlockContext *ctx) = 0;

  virtual void enterAssignment(TensorLangParser::AssignmentContext *ctx) = 0;
  virtual void exitAssignment(TensorLangParser::AssignmentContext *ctx) = 0;

  virtual void enterCrossExpr(TensorLangParser::CrossExprContext *ctx) = 0;
  virtual void exitCrossExpr(TensorLangParser::CrossExprContext *ctx) = 0;

  virtual void enterThetaJoinExpr(TensorLangParser::ThetaJoinExprContext *ctx) = 0;
  virtual void exitThetaJoinExpr(TensorLangParser::ThetaJoinExprContext *ctx) = 0;

  virtual void enterExistentialExpr(TensorLangParser::ExistentialExprContext *ctx) = 0;
  virtual void exitExistentialExpr(TensorLangParser::ExistentialExprContext *ctx) = 0;

  virtual void enterIdExpr(TensorLangParser::IdExprContext *ctx) = 0;
  virtual void exitIdExpr(TensorLangParser::IdExprContext *ctx) = 0;

  virtual void enterScalarExpr(TensorLangParser::ScalarExprContext *ctx) = 0;
  virtual void exitScalarExpr(TensorLangParser::ScalarExprContext *ctx) = 0;

  virtual void enterReduceAllExpr(TensorLangParser::ReduceAllExprContext *ctx) = 0;
  virtual void exitReduceAllExpr(TensorLangParser::ReduceAllExprContext *ctx) = 0;

  virtual void enterSubExpr(TensorLangParser::SubExprContext *ctx) = 0;
  virtual void exitSubExpr(TensorLangParser::SubExprContext *ctx) = 0;

  virtual void enterUniversalExpr(TensorLangParser::UniversalExprContext *ctx) = 0;
  virtual void exitUniversalExpr(TensorLangParser::UniversalExprContext *ctx) = 0;

  virtual void enterAddExpr(TensorLangParser::AddExprContext *ctx) = 0;
  virtual void exitAddExpr(TensorLangParser::AddExprContext *ctx) = 0;

  virtual void enterUnaryExpr(TensorLangParser::UnaryExprContext *ctx) = 0;
  virtual void exitUnaryExpr(TensorLangParser::UnaryExprContext *ctx) = 0;

  virtual void enterPermuteExpr(TensorLangParser::PermuteExprContext *ctx) = 0;
  virtual void exitPermuteExpr(TensorLangParser::PermuteExprContext *ctx) = 0;

  virtual void enterGatherExpr(TensorLangParser::GatherExprContext *ctx) = 0;
  virtual void exitGatherExpr(TensorLangParser::GatherExprContext *ctx) = 0;

  virtual void enterLoadExpr(TensorLangParser::LoadExprContext *ctx) = 0;
  virtual void exitLoadExpr(TensorLangParser::LoadExprContext *ctx) = 0;

  virtual void enterHadamardExpr(TensorLangParser::HadamardExprContext *ctx) = 0;
  virtual void exitHadamardExpr(TensorLangParser::HadamardExprContext *ctx) = 0;

  virtual void enterReduceAxisExpr(TensorLangParser::ReduceAxisExprContext *ctx) = 0;
  virtual void exitReduceAxisExpr(TensorLangParser::ReduceAxisExprContext *ctx) = 0;

  virtual void enterMatMulExpr(TensorLangParser::MatMulExprContext *ctx) = 0;
  virtual void exitMatMulExpr(TensorLangParser::MatMulExprContext *ctx) = 0;

  virtual void enterLambdaPred(TensorLangParser::LambdaPredContext *ctx) = 0;
  virtual void exitLambdaPred(TensorLangParser::LambdaPredContext *ctx) = 0;

  virtual void enterLogicalExpr(TensorLangParser::LogicalExprContext *ctx) = 0;
  virtual void exitLogicalExpr(TensorLangParser::LogicalExprContext *ctx) = 0;

  virtual void enterLogicalAndExpr(TensorLangParser::LogicalAndExprContext *ctx) = 0;
  virtual void exitLogicalAndExpr(TensorLangParser::LogicalAndExprContext *ctx) = 0;

  virtual void enterNotPredicate(TensorLangParser::NotPredicateContext *ctx) = 0;
  virtual void exitNotPredicate(TensorLangParser::NotPredicateContext *ctx) = 0;

  virtual void enterSubPredicate(TensorLangParser::SubPredicateContext *ctx) = 0;
  virtual void exitSubPredicate(TensorLangParser::SubPredicateContext *ctx) = 0;

  virtual void enterCompPredicate(TensorLangParser::CompPredicateContext *ctx) = 0;
  virtual void exitCompPredicate(TensorLangParser::CompPredicateContext *ctx) = 0;

  virtual void enterIntrinsicPredicate(TensorLangParser::IntrinsicPredicateContext *ctx) = 0;
  virtual void exitIntrinsicPredicate(TensorLangParser::IntrinsicPredicateContext *ctx) = 0;

  virtual void enterFloatComparison(TensorLangParser::FloatComparisonContext *ctx) = 0;
  virtual void exitFloatComparison(TensorLangParser::FloatComparisonContext *ctx) = 0;

  virtual void enterFloatExpr(TensorLangParser::FloatExprContext *ctx) = 0;
  virtual void exitFloatExpr(TensorLangParser::FloatExprContext *ctx) = 0;

  virtual void enterFloatTerm(TensorLangParser::FloatTermContext *ctx) = 0;
  virtual void exitFloatTerm(TensorLangParser::FloatTermContext *ctx) = 0;

  virtual void enterAbsFloatExpr(TensorLangParser::AbsFloatExprContext *ctx) = 0;
  virtual void exitAbsFloatExpr(TensorLangParser::AbsFloatExprContext *ctx) = 0;

  virtual void enterCellFloatExpr(TensorLangParser::CellFloatExprContext *ctx) = 0;
  virtual void exitCellFloatExpr(TensorLangParser::CellFloatExprContext *ctx) = 0;

  virtual void enterIdFloatExpr(TensorLangParser::IdFloatExprContext *ctx) = 0;
  virtual void exitIdFloatExpr(TensorLangParser::IdFloatExprContext *ctx) = 0;

  virtual void enterLiteralFloatExpr(TensorLangParser::LiteralFloatExprContext *ctx) = 0;
  virtual void exitLiteralFloatExpr(TensorLangParser::LiteralFloatExprContext *ctx) = 0;

  virtual void enterNegateFloatExpr(TensorLangParser::NegateFloatExprContext *ctx) = 0;
  virtual void exitNegateFloatExpr(TensorLangParser::NegateFloatExprContext *ctx) = 0;


};

