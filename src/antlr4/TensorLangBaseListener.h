
// Generated from src/antlr4/TensorLang.g4 by ANTLR 4.13.2

#pragma once


#include "antlr4-runtime.h"
#include "TensorLangListener.h"


/**
 * This class provides an empty implementation of TensorLangListener,
 * which can be extended to create a listener which only needs to handle a subset
 * of the available methods.
 */
class  TensorLangBaseListener : public TensorLangListener {
public:

  virtual void enterProgram(TensorLangParser::ProgramContext * /*ctx*/) override { }
  virtual void exitProgram(TensorLangParser::ProgramContext * /*ctx*/) override { }

  virtual void enterDeviceConfig(TensorLangParser::DeviceConfigContext * /*ctx*/) override { }
  virtual void exitDeviceConfig(TensorLangParser::DeviceConfigContext * /*ctx*/) override { }

  virtual void enterClusterConfig(TensorLangParser::ClusterConfigContext * /*ctx*/) override { }
  virtual void exitClusterConfig(TensorLangParser::ClusterConfigContext * /*ctx*/) override { }

  virtual void enterOptimizerConfig(TensorLangParser::OptimizerConfigContext * /*ctx*/) override { }
  virtual void exitOptimizerConfig(TensorLangParser::OptimizerConfigContext * /*ctx*/) override { }

  virtual void enterDecayConfig(TensorLangParser::DecayConfigContext * /*ctx*/) override { }
  virtual void exitDecayConfig(TensorLangParser::DecayConfigContext * /*ctx*/) override { }

  virtual void enterStatement(TensorLangParser::StatementContext * /*ctx*/) override { }
  virtual void exitStatement(TensorLangParser::StatementContext * /*ctx*/) override { }

  virtual void enterTensorDeclaration(TensorLangParser::TensorDeclarationContext * /*ctx*/) override { }
  virtual void exitTensorDeclaration(TensorLangParser::TensorDeclarationContext * /*ctx*/) override { }

  virtual void enterTypeID(TensorLangParser::TypeIDContext * /*ctx*/) override { }
  virtual void exitTypeID(TensorLangParser::TypeIDContext * /*ctx*/) override { }

  virtual void enterDatatype(TensorLangParser::DatatypeContext * /*ctx*/) override { }
  virtual void exitDatatype(TensorLangParser::DatatypeContext * /*ctx*/) override { }

  virtual void enterLayout(TensorLangParser::LayoutContext * /*ctx*/) override { }
  virtual void exitLayout(TensorLangParser::LayoutContext * /*ctx*/) override { }

  virtual void enterDimSeq(TensorLangParser::DimSeqContext * /*ctx*/) override { }
  virtual void exitDimSeq(TensorLangParser::DimSeqContext * /*ctx*/) override { }

  virtual void enterDimToken(TensorLangParser::DimTokenContext * /*ctx*/) override { }
  virtual void exitDimToken(TensorLangParser::DimTokenContext * /*ctx*/) override { }

  virtual void enterActiveEpochBlock(TensorLangParser::ActiveEpochBlockContext * /*ctx*/) override { }
  virtual void exitActiveEpochBlock(TensorLangParser::ActiveEpochBlockContext * /*ctx*/) override { }

  virtual void enterAssignment(TensorLangParser::AssignmentContext * /*ctx*/) override { }
  virtual void exitAssignment(TensorLangParser::AssignmentContext * /*ctx*/) override { }

  virtual void enterCrossExpr(TensorLangParser::CrossExprContext * /*ctx*/) override { }
  virtual void exitCrossExpr(TensorLangParser::CrossExprContext * /*ctx*/) override { }

  virtual void enterThetaJoinExpr(TensorLangParser::ThetaJoinExprContext * /*ctx*/) override { }
  virtual void exitThetaJoinExpr(TensorLangParser::ThetaJoinExprContext * /*ctx*/) override { }

  virtual void enterExistentialExpr(TensorLangParser::ExistentialExprContext * /*ctx*/) override { }
  virtual void exitExistentialExpr(TensorLangParser::ExistentialExprContext * /*ctx*/) override { }

  virtual void enterIdExpr(TensorLangParser::IdExprContext * /*ctx*/) override { }
  virtual void exitIdExpr(TensorLangParser::IdExprContext * /*ctx*/) override { }

  virtual void enterScalarExpr(TensorLangParser::ScalarExprContext * /*ctx*/) override { }
  virtual void exitScalarExpr(TensorLangParser::ScalarExprContext * /*ctx*/) override { }

  virtual void enterReduceAllExpr(TensorLangParser::ReduceAllExprContext * /*ctx*/) override { }
  virtual void exitReduceAllExpr(TensorLangParser::ReduceAllExprContext * /*ctx*/) override { }

  virtual void enterSubExpr(TensorLangParser::SubExprContext * /*ctx*/) override { }
  virtual void exitSubExpr(TensorLangParser::SubExprContext * /*ctx*/) override { }

  virtual void enterUniversalExpr(TensorLangParser::UniversalExprContext * /*ctx*/) override { }
  virtual void exitUniversalExpr(TensorLangParser::UniversalExprContext * /*ctx*/) override { }

  virtual void enterAddExpr(TensorLangParser::AddExprContext * /*ctx*/) override { }
  virtual void exitAddExpr(TensorLangParser::AddExprContext * /*ctx*/) override { }

  virtual void enterUnaryExpr(TensorLangParser::UnaryExprContext * /*ctx*/) override { }
  virtual void exitUnaryExpr(TensorLangParser::UnaryExprContext * /*ctx*/) override { }

  virtual void enterPermuteExpr(TensorLangParser::PermuteExprContext * /*ctx*/) override { }
  virtual void exitPermuteExpr(TensorLangParser::PermuteExprContext * /*ctx*/) override { }

  virtual void enterGatherExpr(TensorLangParser::GatherExprContext * /*ctx*/) override { }
  virtual void exitGatherExpr(TensorLangParser::GatherExprContext * /*ctx*/) override { }

  virtual void enterLoadExpr(TensorLangParser::LoadExprContext * /*ctx*/) override { }
  virtual void exitLoadExpr(TensorLangParser::LoadExprContext * /*ctx*/) override { }

  virtual void enterHadamardExpr(TensorLangParser::HadamardExprContext * /*ctx*/) override { }
  virtual void exitHadamardExpr(TensorLangParser::HadamardExprContext * /*ctx*/) override { }

  virtual void enterReduceAxisExpr(TensorLangParser::ReduceAxisExprContext * /*ctx*/) override { }
  virtual void exitReduceAxisExpr(TensorLangParser::ReduceAxisExprContext * /*ctx*/) override { }

  virtual void enterMatMulExpr(TensorLangParser::MatMulExprContext * /*ctx*/) override { }
  virtual void exitMatMulExpr(TensorLangParser::MatMulExprContext * /*ctx*/) override { }

  virtual void enterLambdaPred(TensorLangParser::LambdaPredContext * /*ctx*/) override { }
  virtual void exitLambdaPred(TensorLangParser::LambdaPredContext * /*ctx*/) override { }

  virtual void enterLogicalExpr(TensorLangParser::LogicalExprContext * /*ctx*/) override { }
  virtual void exitLogicalExpr(TensorLangParser::LogicalExprContext * /*ctx*/) override { }

  virtual void enterLogicalAndExpr(TensorLangParser::LogicalAndExprContext * /*ctx*/) override { }
  virtual void exitLogicalAndExpr(TensorLangParser::LogicalAndExprContext * /*ctx*/) override { }

  virtual void enterNotPredicate(TensorLangParser::NotPredicateContext * /*ctx*/) override { }
  virtual void exitNotPredicate(TensorLangParser::NotPredicateContext * /*ctx*/) override { }

  virtual void enterSubPredicate(TensorLangParser::SubPredicateContext * /*ctx*/) override { }
  virtual void exitSubPredicate(TensorLangParser::SubPredicateContext * /*ctx*/) override { }

  virtual void enterCompPredicate(TensorLangParser::CompPredicateContext * /*ctx*/) override { }
  virtual void exitCompPredicate(TensorLangParser::CompPredicateContext * /*ctx*/) override { }

  virtual void enterIntrinsicPredicate(TensorLangParser::IntrinsicPredicateContext * /*ctx*/) override { }
  virtual void exitIntrinsicPredicate(TensorLangParser::IntrinsicPredicateContext * /*ctx*/) override { }

  virtual void enterFloatComparison(TensorLangParser::FloatComparisonContext * /*ctx*/) override { }
  virtual void exitFloatComparison(TensorLangParser::FloatComparisonContext * /*ctx*/) override { }

  virtual void enterFloatExpr(TensorLangParser::FloatExprContext * /*ctx*/) override { }
  virtual void exitFloatExpr(TensorLangParser::FloatExprContext * /*ctx*/) override { }

  virtual void enterFloatTerm(TensorLangParser::FloatTermContext * /*ctx*/) override { }
  virtual void exitFloatTerm(TensorLangParser::FloatTermContext * /*ctx*/) override { }

  virtual void enterAbsFloatExpr(TensorLangParser::AbsFloatExprContext * /*ctx*/) override { }
  virtual void exitAbsFloatExpr(TensorLangParser::AbsFloatExprContext * /*ctx*/) override { }

  virtual void enterCellFloatExpr(TensorLangParser::CellFloatExprContext * /*ctx*/) override { }
  virtual void exitCellFloatExpr(TensorLangParser::CellFloatExprContext * /*ctx*/) override { }

  virtual void enterIdFloatExpr(TensorLangParser::IdFloatExprContext * /*ctx*/) override { }
  virtual void exitIdFloatExpr(TensorLangParser::IdFloatExprContext * /*ctx*/) override { }

  virtual void enterLiteralFloatExpr(TensorLangParser::LiteralFloatExprContext * /*ctx*/) override { }
  virtual void exitLiteralFloatExpr(TensorLangParser::LiteralFloatExprContext * /*ctx*/) override { }

  virtual void enterNegateFloatExpr(TensorLangParser::NegateFloatExprContext * /*ctx*/) override { }
  virtual void exitNegateFloatExpr(TensorLangParser::NegateFloatExprContext * /*ctx*/) override { }


  virtual void enterEveryRule(antlr4::ParserRuleContext * /*ctx*/) override { }
  virtual void exitEveryRule(antlr4::ParserRuleContext * /*ctx*/) override { }
  virtual void visitTerminal(antlr4::tree::TerminalNode * /*node*/) override { }
  virtual void visitErrorNode(antlr4::tree::ErrorNode * /*node*/) override { }

};

