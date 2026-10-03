
// Generated from src/antlr4/TensorLang.g4 by ANTLR 4.13.2

#pragma once


#include "antlr4-runtime.h"




class  TensorLangParser : public antlr4::Parser {
public:
  enum {
    T__0 = 1, T__1 = 2, T__2 = 3, T__3 = 4, T__4 = 5, T__5 = 6, T__6 = 7, 
    T__7 = 8, T__8 = 9, T__9 = 10, T__10 = 11, T__11 = 12, T__12 = 13, T__13 = 14, 
    T__14 = 15, T__15 = 16, T__16 = 17, T__17 = 18, T__18 = 19, T__19 = 20, 
    T__20 = 21, T__21 = 22, T__22 = 23, T__23 = 24, T__24 = 25, T__25 = 26, 
    T__26 = 27, T__27 = 28, T__28 = 29, T__29 = 30, T__30 = 31, T__31 = 32, 
    T__32 = 33, T__33 = 34, T__34 = 35, T__35 = 36, T__36 = 37, T__37 = 38, 
    T__38 = 39, T__39 = 40, T__40 = 41, T__41 = 42, T__42 = 43, T__43 = 44, 
    DEVICE = 45, BOOLEAN = 46, OPT_TYPE = 47, DECAY_TYPE = 48, CELL_OP = 49, 
    COMP_OP = 50, INTRINSIC_OP = 51, INT = 52, FLOAT = 53, ID = 54, STRING = 55, 
    WS = 56
  };

  enum {
    RuleProgram = 0, RuleConfigDirective = 1, RuleStatement = 2, RuleTensorDeclaration = 3, 
    RuleTypeID = 4, RuleDatatype = 5, RuleLayout = 6, RuleDimSeq = 7, RuleDimToken = 8, 
    RuleActiveEpochBlock = 9, RuleAssignment = 10, RuleExpr = 11, RuleLambdaPred = 12, 
    RuleLogicalExpr = 13, RuleLogicalAndExpr = 14, RuleLogicalNotExpr = 15, 
    RuleFloatComparison = 16, RuleFloatExpr = 17, RuleFloatTerm = 18, RuleFloatFactor = 19
  };

  explicit TensorLangParser(antlr4::TokenStream *input);

  TensorLangParser(antlr4::TokenStream *input, const antlr4::atn::ParserATNSimulatorOptions &options);

  ~TensorLangParser() override;

  std::string getGrammarFileName() const override;

  const antlr4::atn::ATN& getATN() const override;

  const std::vector<std::string>& getRuleNames() const override;

  const antlr4::dfa::Vocabulary& getVocabulary() const override;

  antlr4::atn::SerializedATNView getSerializedATN() const override;


  class ProgramContext;
  class ConfigDirectiveContext;
  class StatementContext;
  class TensorDeclarationContext;
  class TypeIDContext;
  class DatatypeContext;
  class LayoutContext;
  class DimSeqContext;
  class DimTokenContext;
  class ActiveEpochBlockContext;
  class AssignmentContext;
  class ExprContext;
  class LambdaPredContext;
  class LogicalExprContext;
  class LogicalAndExprContext;
  class LogicalNotExprContext;
  class FloatComparisonContext;
  class FloatExprContext;
  class FloatTermContext;
  class FloatFactorContext; 

  class  ProgramContext : public antlr4::ParserRuleContext {
  public:
    ProgramContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *EOF();
    std::vector<ConfigDirectiveContext *> configDirective();
    ConfigDirectiveContext* configDirective(size_t i);
    std::vector<StatementContext *> statement();
    StatementContext* statement(size_t i);

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  ProgramContext* program();

  class  ConfigDirectiveContext : public antlr4::ParserRuleContext {
  public:
    ConfigDirectiveContext(antlr4::ParserRuleContext *parent, size_t invokingState);
   
    ConfigDirectiveContext() = default;
    void copyFrom(ConfigDirectiveContext *context);
    using antlr4::ParserRuleContext::copyFrom;

    virtual size_t getRuleIndex() const override;

   
  };

  class  OptimizerConfigContext : public ConfigDirectiveContext {
  public:
    OptimizerConfigContext(ConfigDirectiveContext *ctx);

    antlr4::tree::TerminalNode *OPT_TYPE();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  DecayConfigContext : public ConfigDirectiveContext {
  public:
    DecayConfigContext(ConfigDirectiveContext *ctx);

    antlr4::tree::TerminalNode *DECAY_TYPE();
    antlr4::tree::TerminalNode *FLOAT();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  DeviceConfigContext : public ConfigDirectiveContext {
  public:
    DeviceConfigContext(ConfigDirectiveContext *ctx);

    antlr4::tree::TerminalNode *DEVICE();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  ClusterConfigContext : public ConfigDirectiveContext {
  public:
    ClusterConfigContext(ConfigDirectiveContext *ctx);

    antlr4::tree::TerminalNode *BOOLEAN();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  ConfigDirectiveContext* configDirective();

  class  StatementContext : public antlr4::ParserRuleContext {
  public:
    StatementContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    TensorDeclarationContext *tensorDeclaration();
    ActiveEpochBlockContext *activeEpochBlock();
    AssignmentContext *assignment();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  StatementContext* statement();

  class  TensorDeclarationContext : public antlr4::ParserRuleContext {
  public:
    TensorDeclarationContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    TypeIDContext *typeID();
    antlr4::tree::TerminalNode *ID();
    ExprContext *expr();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  TensorDeclarationContext* tensorDeclaration();

  class  TypeIDContext : public antlr4::ParserRuleContext {
  public:
    TypeIDContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    DatatypeContext *datatype();
    LayoutContext *layout();
    DimSeqContext *dimSeq();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  TypeIDContext* typeID();

  class  DatatypeContext : public antlr4::ParserRuleContext {
  public:
    DatatypeContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  DatatypeContext* datatype();

  class  LayoutContext : public antlr4::ParserRuleContext {
  public:
    LayoutContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  LayoutContext* layout();

  class  DimSeqContext : public antlr4::ParserRuleContext {
  public:
    DimSeqContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    std::vector<DimTokenContext *> dimToken();
    DimTokenContext* dimToken(size_t i);

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  DimSeqContext* dimSeq();

  class  DimTokenContext : public antlr4::ParserRuleContext {
  public:
    DimTokenContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *INT();
    antlr4::tree::TerminalNode *ID();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  DimTokenContext* dimToken();

  class  ActiveEpochBlockContext : public antlr4::ParserRuleContext {
  public:
    ActiveEpochBlockContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *INT();
    antlr4::tree::TerminalNode *ID();
    antlr4::tree::TerminalNode *FLOAT();
    std::vector<StatementContext *> statement();
    StatementContext* statement(size_t i);

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  ActiveEpochBlockContext* activeEpochBlock();

  class  AssignmentContext : public antlr4::ParserRuleContext {
  public:
    AssignmentContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *ID();
    ExprContext *expr();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  AssignmentContext* assignment();

  class  ExprContext : public antlr4::ParserRuleContext {
  public:
    ExprContext(antlr4::ParserRuleContext *parent, size_t invokingState);
   
    ExprContext() = default;
    void copyFrom(ExprContext *context);
    using antlr4::ParserRuleContext::copyFrom;

    virtual size_t getRuleIndex() const override;

   
  };

  class  CrossExprContext : public ExprContext {
  public:
    CrossExprContext(ExprContext *ctx);

    std::vector<ExprContext *> expr();
    ExprContext* expr(size_t i);
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  ThetaJoinExprContext : public ExprContext {
  public:
    ThetaJoinExprContext(ExprContext *ctx);

    std::vector<ExprContext *> expr();
    ExprContext* expr(size_t i);
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  ExistentialExprContext : public ExprContext {
  public:
    ExistentialExprContext(ExprContext *ctx);

    ExprContext *expr();
    DimSeqContext *dimSeq();
    LambdaPredContext *lambdaPred();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  IdExprContext : public ExprContext {
  public:
    IdExprContext(ExprContext *ctx);

    antlr4::tree::TerminalNode *ID();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  ScalarExprContext : public ExprContext {
  public:
    ScalarExprContext(ExprContext *ctx);

    TypeIDContext *typeID();
    antlr4::tree::TerminalNode *FLOAT();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  ReduceAllExprContext : public ExprContext {
  public:
    ReduceAllExprContext(ExprContext *ctx);

    ExprContext *expr();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  SubExprContext : public ExprContext {
  public:
    SubExprContext(ExprContext *ctx);

    std::vector<ExprContext *> expr();
    ExprContext* expr(size_t i);
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  UniversalExprContext : public ExprContext {
  public:
    UniversalExprContext(ExprContext *ctx);

    ExprContext *expr();
    DimSeqContext *dimSeq();
    LambdaPredContext *lambdaPred();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  AddExprContext : public ExprContext {
  public:
    AddExprContext(ExprContext *ctx);

    std::vector<ExprContext *> expr();
    ExprContext* expr(size_t i);
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  UnaryExprContext : public ExprContext {
  public:
    UnaryExprContext(ExprContext *ctx);

    ExprContext *expr();
    antlr4::tree::TerminalNode *CELL_OP();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  PermuteExprContext : public ExprContext {
  public:
    PermuteExprContext(ExprContext *ctx);

    ExprContext *expr();
    DimSeqContext *dimSeq();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  GatherExprContext : public ExprContext {
  public:
    GatherExprContext(ExprContext *ctx);

    std::vector<ExprContext *> expr();
    ExprContext* expr(size_t i);
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  LoadExprContext : public ExprContext {
  public:
    LoadExprContext(ExprContext *ctx);

    std::vector<antlr4::tree::TerminalNode *> STRING();
    antlr4::tree::TerminalNode* STRING(size_t i);
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  HadamardExprContext : public ExprContext {
  public:
    HadamardExprContext(ExprContext *ctx);

    std::vector<ExprContext *> expr();
    ExprContext* expr(size_t i);
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  ReduceAxisExprContext : public ExprContext {
  public:
    ReduceAxisExprContext(ExprContext *ctx);

    ExprContext *expr();
    antlr4::tree::TerminalNode *INT();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  MatMulExprContext : public ExprContext {
  public:
    MatMulExprContext(ExprContext *ctx);

    std::vector<ExprContext *> expr();
    ExprContext* expr(size_t i);
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  ExprContext* expr();
  ExprContext* expr(int precedence);
  class  LambdaPredContext : public antlr4::ParserRuleContext {
  public:
    LambdaPredContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *ID();
    LogicalExprContext *logicalExpr();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  LambdaPredContext* lambdaPred();

  class  LogicalExprContext : public antlr4::ParserRuleContext {
  public:
    LogicalExprContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    std::vector<LogicalAndExprContext *> logicalAndExpr();
    LogicalAndExprContext* logicalAndExpr(size_t i);

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  LogicalExprContext* logicalExpr();

  class  LogicalAndExprContext : public antlr4::ParserRuleContext {
  public:
    LogicalAndExprContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    std::vector<LogicalNotExprContext *> logicalNotExpr();
    LogicalNotExprContext* logicalNotExpr(size_t i);

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  LogicalAndExprContext* logicalAndExpr();

  class  LogicalNotExprContext : public antlr4::ParserRuleContext {
  public:
    LogicalNotExprContext(antlr4::ParserRuleContext *parent, size_t invokingState);
   
    LogicalNotExprContext() = default;
    void copyFrom(LogicalNotExprContext *context);
    using antlr4::ParserRuleContext::copyFrom;

    virtual size_t getRuleIndex() const override;

   
  };

  class  IntrinsicPredicateContext : public LogicalNotExprContext {
  public:
    IntrinsicPredicateContext(LogicalNotExprContext *ctx);

    antlr4::tree::TerminalNode *INTRINSIC_OP();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  SubPredicateContext : public LogicalNotExprContext {
  public:
    SubPredicateContext(LogicalNotExprContext *ctx);

    LogicalExprContext *logicalExpr();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  CompPredicateContext : public LogicalNotExprContext {
  public:
    CompPredicateContext(LogicalNotExprContext *ctx);

    FloatComparisonContext *floatComparison();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  NotPredicateContext : public LogicalNotExprContext {
  public:
    NotPredicateContext(LogicalNotExprContext *ctx);

    LogicalNotExprContext *logicalNotExpr();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  LogicalNotExprContext* logicalNotExpr();

  class  FloatComparisonContext : public antlr4::ParserRuleContext {
  public:
    FloatComparisonContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    std::vector<FloatExprContext *> floatExpr();
    FloatExprContext* floatExpr(size_t i);
    antlr4::tree::TerminalNode *COMP_OP();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  FloatComparisonContext* floatComparison();

  class  FloatExprContext : public antlr4::ParserRuleContext {
  public:
    FloatExprContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    std::vector<FloatTermContext *> floatTerm();
    FloatTermContext* floatTerm(size_t i);

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  FloatExprContext* floatExpr();

  class  FloatTermContext : public antlr4::ParserRuleContext {
  public:
    FloatTermContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    std::vector<FloatFactorContext *> floatFactor();
    FloatFactorContext* floatFactor(size_t i);

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  FloatTermContext* floatTerm();

  class  FloatFactorContext : public antlr4::ParserRuleContext {
  public:
    FloatFactorContext(antlr4::ParserRuleContext *parent, size_t invokingState);
   
    FloatFactorContext() = default;
    void copyFrom(FloatFactorContext *context);
    using antlr4::ParserRuleContext::copyFrom;

    virtual size_t getRuleIndex() const override;

   
  };

  class  NegateFloatExprContext : public FloatFactorContext {
  public:
    NegateFloatExprContext(FloatFactorContext *ctx);

    FloatFactorContext *floatFactor();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  CellFloatExprContext : public FloatFactorContext {
  public:
    CellFloatExprContext(FloatFactorContext *ctx);

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  AbsFloatExprContext : public FloatFactorContext {
  public:
    AbsFloatExprContext(FloatFactorContext *ctx);

    FloatExprContext *floatExpr();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  IdFloatExprContext : public FloatFactorContext {
  public:
    IdFloatExprContext(FloatFactorContext *ctx);

    antlr4::tree::TerminalNode *ID();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  LiteralFloatExprContext : public FloatFactorContext {
  public:
    LiteralFloatExprContext(FloatFactorContext *ctx);

    antlr4::tree::TerminalNode *FLOAT();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  FloatFactorContext* floatFactor();


  bool sempred(antlr4::RuleContext *_localctx, size_t ruleIndex, size_t predicateIndex) override;

  bool exprSempred(ExprContext *_localctx, size_t predicateIndex);

  // By default the static state used to implement the parser is lazily initialized during the first
  // call to the constructor. You can call this function if you wish to initialize the static state
  // ahead of time.
  static void initialize();

private:
};

