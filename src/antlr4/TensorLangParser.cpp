
// Generated from src/antlr4/TensorLang.g4 by ANTLR 4.13.2


#include "TensorLangListener.h"
#include "TensorLangVisitor.h"

#include "TensorLangParser.h"


using namespace antlrcpp;

using namespace antlr4;

namespace {

struct TensorLangParserStaticData final {
  TensorLangParserStaticData(std::vector<std::string> ruleNames,
                        std::vector<std::string> literalNames,
                        std::vector<std::string> symbolicNames)
      : ruleNames(std::move(ruleNames)), literalNames(std::move(literalNames)),
        symbolicNames(std::move(symbolicNames)),
        vocabulary(this->literalNames, this->symbolicNames) {}

  TensorLangParserStaticData(const TensorLangParserStaticData&) = delete;
  TensorLangParserStaticData(TensorLangParserStaticData&&) = delete;
  TensorLangParserStaticData& operator=(const TensorLangParserStaticData&) = delete;
  TensorLangParserStaticData& operator=(TensorLangParserStaticData&&) = delete;

  std::vector<antlr4::dfa::DFA> decisionToDFA;
  antlr4::atn::PredictionContextCache sharedContextCache;
  const std::vector<std::string> ruleNames;
  const std::vector<std::string> literalNames;
  const std::vector<std::string> symbolicNames;
  const antlr4::dfa::Vocabulary vocabulary;
  antlr4::atn::SerializedATNView serializedATN;
  std::unique_ptr<antlr4::atn::ATN> atn;
};

::antlr4::internal::OnceFlag tensorlangParserOnceFlag;
#if ANTLR4_USE_THREAD_LOCAL_CACHE
static thread_local
#endif
std::unique_ptr<TensorLangParserStaticData> tensorlangParserStaticData = nullptr;

void tensorlangParserInitialize() {
#if ANTLR4_USE_THREAD_LOCAL_CACHE
  if (tensorlangParserStaticData != nullptr) {
    return;
  }
#else
  assert(tensorlangParserStaticData == nullptr);
#endif
  auto staticData = std::make_unique<TensorLangParserStaticData>(
    std::vector<std::string>{
      "program", "configDirective", "statement", "tensorDeclaration", "typeID", 
      "datatype", "layout", "dimSeq", "dimToken", "activeEpochBlock", "assignment", 
      "expr", "lambdaPred", "logicalExpr", "logicalAndExpr", "logicalNotExpr", 
      "floatComparison", "floatExpr", "floatTerm", "floatFactor"
    },
    std::vector<std::string>{
      "", "'use'", "'device'", "';'", "'cluster'", "'optimizer'", "'decay'", 
      "'='", "'MetaTensor'", "'<'", "','", "'>'", "'float'", "'double'", 
      "'int'", "'Dense'", "'SparseCOO'", "'epoch_loop'", "'('", "')'", "'{'", 
      "'}'", "'load_safetensors'", "'from_scalar'", "'*'", "'+'", "'-'", 
      "'%'", "'.element_wise_mul'", "'.apply'", "'()'", "'.reduce_all_sum()'", 
      "'.reduce_sum'", "'.permute_axes'", "'.gather_nd'", "'.tensor_theta_join'", 
      "'.evaluate_existential'", "'.evaluate_universal'", "'->'", "'||'", 
      "'&&'", "'!'", "'/'", "'abs'", "'cell'"
    },
    std::vector<std::string>{
      "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", 
      "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", 
      "", "", "", "", "", "", "", "", "", "", "", "DEVICE", "BOOLEAN", "OPT_TYPE", 
      "DECAY_TYPE", "CELL_OP", "COMP_OP", "INTRINSIC_OP", "INT", "FLOAT", 
      "ID", "STRING", "WS"
    }
  );
  static const int32_t serializedATNSegment[] = {
  	4,1,56,284,2,0,7,0,2,1,7,1,2,2,7,2,2,3,7,3,2,4,7,4,2,5,7,5,2,6,7,6,2,
  	7,7,7,2,8,7,8,2,9,7,9,2,10,7,10,2,11,7,11,2,12,7,12,2,13,7,13,2,14,7,
  	14,2,15,7,15,2,16,7,16,2,17,7,17,2,18,7,18,2,19,7,19,1,0,5,0,42,8,0,10,
  	0,12,0,45,9,0,1,0,4,0,48,8,0,11,0,12,0,49,1,0,1,0,1,1,1,1,1,1,1,1,1,1,
  	1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,3,1,71,8,1,1,2,1,2,1,
  	2,3,2,76,8,2,1,3,1,3,1,3,1,3,1,3,1,3,1,4,1,4,1,4,1,4,1,4,1,4,1,4,1,4,
  	1,4,1,5,1,5,1,6,1,6,1,7,1,7,1,7,5,7,100,8,7,10,7,12,7,103,9,7,1,8,1,8,
  	1,9,1,9,1,9,1,9,1,9,1,9,1,9,1,9,1,9,1,9,4,9,117,8,9,11,9,12,9,118,1,9,
  	1,9,1,10,1,10,1,10,1,10,1,10,1,11,1,11,1,11,1,11,1,11,1,11,1,11,1,11,
  	1,11,1,11,1,11,1,11,1,11,1,11,1,11,1,11,3,11,144,8,11,1,11,1,11,1,11,
  	1,11,1,11,1,11,1,11,1,11,1,11,1,11,1,11,1,11,1,11,1,11,1,11,1,11,1,11,
  	1,11,1,11,1,11,1,11,1,11,1,11,1,11,1,11,1,11,1,11,1,11,1,11,1,11,1,11,
  	1,11,1,11,1,11,1,11,1,11,1,11,1,11,1,11,1,11,1,11,1,11,1,11,1,11,1,11,
  	1,11,1,11,1,11,1,11,1,11,1,11,1,11,1,11,1,11,1,11,1,11,1,11,1,11,1,11,
  	1,11,1,11,1,11,1,11,1,11,1,11,1,11,1,11,1,11,1,11,5,11,215,8,11,10,11,
  	12,11,218,9,11,1,12,1,12,1,12,1,12,1,12,1,12,1,13,1,13,1,13,5,13,229,
  	8,13,10,13,12,13,232,9,13,1,14,1,14,1,14,5,14,237,8,14,10,14,12,14,240,
  	9,14,1,15,1,15,1,15,1,15,1,15,1,15,1,15,1,15,3,15,250,8,15,1,16,1,16,
  	1,16,1,16,1,17,1,17,1,17,5,17,259,8,17,10,17,12,17,262,9,17,1,18,1,18,
  	1,18,5,18,267,8,18,10,18,12,18,270,9,18,1,19,1,19,1,19,1,19,1,19,1,19,
  	1,19,1,19,1,19,1,19,3,19,282,8,19,1,19,0,1,22,20,0,2,4,6,8,10,12,14,16,
  	18,20,22,24,26,28,30,32,34,36,38,0,5,1,0,12,14,1,0,15,16,2,0,52,52,54,
  	54,1,0,25,26,2,0,24,24,42,42,298,0,43,1,0,0,0,2,70,1,0,0,0,4,75,1,0,0,
  	0,6,77,1,0,0,0,8,83,1,0,0,0,10,92,1,0,0,0,12,94,1,0,0,0,14,96,1,0,0,0,
  	16,104,1,0,0,0,18,106,1,0,0,0,20,122,1,0,0,0,22,143,1,0,0,0,24,219,1,
  	0,0,0,26,225,1,0,0,0,28,233,1,0,0,0,30,249,1,0,0,0,32,251,1,0,0,0,34,
  	255,1,0,0,0,36,263,1,0,0,0,38,281,1,0,0,0,40,42,3,2,1,0,41,40,1,0,0,0,
  	42,45,1,0,0,0,43,41,1,0,0,0,43,44,1,0,0,0,44,47,1,0,0,0,45,43,1,0,0,0,
  	46,48,3,4,2,0,47,46,1,0,0,0,48,49,1,0,0,0,49,47,1,0,0,0,49,50,1,0,0,0,
  	50,51,1,0,0,0,51,52,5,0,0,1,52,1,1,0,0,0,53,54,5,1,0,0,54,55,5,2,0,0,
  	55,56,5,45,0,0,56,71,5,3,0,0,57,58,5,1,0,0,58,59,5,4,0,0,59,60,5,46,0,
  	0,60,71,5,3,0,0,61,62,5,1,0,0,62,63,5,5,0,0,63,64,5,47,0,0,64,71,5,3,
  	0,0,65,66,5,1,0,0,66,67,5,6,0,0,67,68,5,48,0,0,68,69,5,53,0,0,69,71,5,
  	3,0,0,70,53,1,0,0,0,70,57,1,0,0,0,70,61,1,0,0,0,70,65,1,0,0,0,71,3,1,
  	0,0,0,72,76,3,6,3,0,73,76,3,18,9,0,74,76,3,20,10,0,75,72,1,0,0,0,75,73,
  	1,0,0,0,75,74,1,0,0,0,76,5,1,0,0,0,77,78,3,8,4,0,78,79,5,54,0,0,79,80,
  	5,7,0,0,80,81,3,22,11,0,81,82,5,3,0,0,82,7,1,0,0,0,83,84,5,8,0,0,84,85,
  	5,9,0,0,85,86,3,10,5,0,86,87,5,10,0,0,87,88,3,12,6,0,88,89,5,10,0,0,89,
  	90,3,14,7,0,90,91,5,11,0,0,91,9,1,0,0,0,92,93,7,0,0,0,93,11,1,0,0,0,94,
  	95,7,1,0,0,95,13,1,0,0,0,96,101,3,16,8,0,97,98,5,10,0,0,98,100,3,16,8,
  	0,99,97,1,0,0,0,100,103,1,0,0,0,101,99,1,0,0,0,101,102,1,0,0,0,102,15,
  	1,0,0,0,103,101,1,0,0,0,104,105,7,2,0,0,105,17,1,0,0,0,106,107,5,17,0,
  	0,107,108,5,18,0,0,108,109,5,52,0,0,109,110,5,10,0,0,110,111,5,54,0,0,
  	111,112,5,10,0,0,112,113,5,53,0,0,113,114,5,19,0,0,114,116,5,20,0,0,115,
  	117,3,4,2,0,116,115,1,0,0,0,117,118,1,0,0,0,118,116,1,0,0,0,118,119,1,
  	0,0,0,119,120,1,0,0,0,120,121,5,21,0,0,121,19,1,0,0,0,122,123,5,54,0,
  	0,123,124,5,7,0,0,124,125,3,22,11,0,125,126,5,3,0,0,126,21,1,0,0,0,127,
  	128,6,11,-1,0,128,144,5,54,0,0,129,130,5,22,0,0,130,131,5,18,0,0,131,
  	132,5,55,0,0,132,133,5,10,0,0,133,134,5,55,0,0,134,144,5,19,0,0,135,136,
  	5,23,0,0,136,137,5,9,0,0,137,138,3,8,4,0,138,139,5,11,0,0,139,140,5,18,
  	0,0,140,141,5,53,0,0,141,142,5,19,0,0,142,144,1,0,0,0,143,127,1,0,0,0,
  	143,129,1,0,0,0,143,135,1,0,0,0,144,216,1,0,0,0,145,146,10,13,0,0,146,
  	147,5,24,0,0,147,215,3,22,11,14,148,149,10,12,0,0,149,150,5,25,0,0,150,
  	215,3,22,11,13,151,152,10,11,0,0,152,153,5,26,0,0,153,215,3,22,11,12,
  	154,155,10,10,0,0,155,156,5,27,0,0,156,215,3,22,11,11,157,158,10,9,0,
  	0,158,159,5,28,0,0,159,160,5,18,0,0,160,161,3,22,11,0,161,162,5,19,0,
  	0,162,215,1,0,0,0,163,164,10,8,0,0,164,165,5,29,0,0,165,166,5,9,0,0,166,
  	167,5,49,0,0,167,168,5,11,0,0,168,215,5,30,0,0,169,170,10,7,0,0,170,215,
  	5,31,0,0,171,172,10,6,0,0,172,173,5,32,0,0,173,174,5,9,0,0,174,175,5,
  	52,0,0,175,176,5,11,0,0,176,215,5,30,0,0,177,178,10,5,0,0,178,179,5,33,
  	0,0,179,180,5,9,0,0,180,181,3,14,7,0,181,182,5,11,0,0,182,183,5,30,0,
  	0,183,215,1,0,0,0,184,185,10,4,0,0,185,186,5,34,0,0,186,187,5,18,0,0,
  	187,188,3,22,11,0,188,189,5,19,0,0,189,215,1,0,0,0,190,191,10,3,0,0,191,
  	192,5,35,0,0,192,193,5,18,0,0,193,194,3,22,11,0,194,195,5,19,0,0,195,
  	215,1,0,0,0,196,197,10,2,0,0,197,198,5,36,0,0,198,199,5,9,0,0,199,200,
  	3,14,7,0,200,201,5,11,0,0,201,202,5,18,0,0,202,203,3,24,12,0,203,204,
  	5,19,0,0,204,215,1,0,0,0,205,206,10,1,0,0,206,207,5,37,0,0,207,208,5,
  	9,0,0,208,209,3,14,7,0,209,210,5,11,0,0,210,211,5,18,0,0,211,212,3,24,
  	12,0,212,213,5,19,0,0,213,215,1,0,0,0,214,145,1,0,0,0,214,148,1,0,0,0,
  	214,151,1,0,0,0,214,154,1,0,0,0,214,157,1,0,0,0,214,163,1,0,0,0,214,169,
  	1,0,0,0,214,171,1,0,0,0,214,177,1,0,0,0,214,184,1,0,0,0,214,190,1,0,0,
  	0,214,196,1,0,0,0,214,205,1,0,0,0,215,218,1,0,0,0,216,214,1,0,0,0,216,
  	217,1,0,0,0,217,23,1,0,0,0,218,216,1,0,0,0,219,220,5,18,0,0,220,221,5,
  	54,0,0,221,222,5,19,0,0,222,223,5,38,0,0,223,224,3,26,13,0,224,25,1,0,
  	0,0,225,230,3,28,14,0,226,227,5,39,0,0,227,229,3,28,14,0,228,226,1,0,
  	0,0,229,232,1,0,0,0,230,228,1,0,0,0,230,231,1,0,0,0,231,27,1,0,0,0,232,
  	230,1,0,0,0,233,238,3,30,15,0,234,235,5,40,0,0,235,237,3,30,15,0,236,
  	234,1,0,0,0,237,240,1,0,0,0,238,236,1,0,0,0,238,239,1,0,0,0,239,29,1,
  	0,0,0,240,238,1,0,0,0,241,242,5,41,0,0,242,250,3,30,15,0,243,244,5,18,
  	0,0,244,245,3,26,13,0,245,246,5,19,0,0,246,250,1,0,0,0,247,250,3,32,16,
  	0,248,250,5,51,0,0,249,241,1,0,0,0,249,243,1,0,0,0,249,247,1,0,0,0,249,
  	248,1,0,0,0,250,31,1,0,0,0,251,252,3,34,17,0,252,253,5,50,0,0,253,254,
  	3,34,17,0,254,33,1,0,0,0,255,260,3,36,18,0,256,257,7,3,0,0,257,259,3,
  	36,18,0,258,256,1,0,0,0,259,262,1,0,0,0,260,258,1,0,0,0,260,261,1,0,0,
  	0,261,35,1,0,0,0,262,260,1,0,0,0,263,268,3,38,19,0,264,265,7,4,0,0,265,
  	267,3,38,19,0,266,264,1,0,0,0,267,270,1,0,0,0,268,266,1,0,0,0,268,269,
  	1,0,0,0,269,37,1,0,0,0,270,268,1,0,0,0,271,272,5,43,0,0,272,273,5,18,
  	0,0,273,274,3,34,17,0,274,275,5,19,0,0,275,282,1,0,0,0,276,282,5,44,0,
  	0,277,282,5,54,0,0,278,282,5,53,0,0,279,280,5,26,0,0,280,282,3,38,19,
  	0,281,271,1,0,0,0,281,276,1,0,0,0,281,277,1,0,0,0,281,278,1,0,0,0,281,
  	279,1,0,0,0,282,39,1,0,0,0,15,43,49,70,75,101,118,143,214,216,230,238,
  	249,260,268,281
  };
  staticData->serializedATN = antlr4::atn::SerializedATNView(serializedATNSegment, sizeof(serializedATNSegment) / sizeof(serializedATNSegment[0]));

  antlr4::atn::ATNDeserializer deserializer;
  staticData->atn = deserializer.deserialize(staticData->serializedATN);

  const size_t count = staticData->atn->getNumberOfDecisions();
  staticData->decisionToDFA.reserve(count);
  for (size_t i = 0; i < count; i++) { 
    staticData->decisionToDFA.emplace_back(staticData->atn->getDecisionState(i), i);
  }
  tensorlangParserStaticData = std::move(staticData);
}

}

TensorLangParser::TensorLangParser(TokenStream *input) : TensorLangParser(input, antlr4::atn::ParserATNSimulatorOptions()) {}

TensorLangParser::TensorLangParser(TokenStream *input, const antlr4::atn::ParserATNSimulatorOptions &options) : Parser(input) {
  TensorLangParser::initialize();
  _interpreter = new atn::ParserATNSimulator(this, *tensorlangParserStaticData->atn, tensorlangParserStaticData->decisionToDFA, tensorlangParserStaticData->sharedContextCache, options);
}

TensorLangParser::~TensorLangParser() {
  delete _interpreter;
}

const atn::ATN& TensorLangParser::getATN() const {
  return *tensorlangParserStaticData->atn;
}

std::string TensorLangParser::getGrammarFileName() const {
  return "TensorLang.g4";
}

const std::vector<std::string>& TensorLangParser::getRuleNames() const {
  return tensorlangParserStaticData->ruleNames;
}

const dfa::Vocabulary& TensorLangParser::getVocabulary() const {
  return tensorlangParserStaticData->vocabulary;
}

antlr4::atn::SerializedATNView TensorLangParser::getSerializedATN() const {
  return tensorlangParserStaticData->serializedATN;
}


//----------------- ProgramContext ------------------------------------------------------------------

TensorLangParser::ProgramContext::ProgramContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* TensorLangParser::ProgramContext::EOF() {
  return getToken(TensorLangParser::EOF, 0);
}

std::vector<TensorLangParser::ConfigDirectiveContext *> TensorLangParser::ProgramContext::configDirective() {
  return getRuleContexts<TensorLangParser::ConfigDirectiveContext>();
}

TensorLangParser::ConfigDirectiveContext* TensorLangParser::ProgramContext::configDirective(size_t i) {
  return getRuleContext<TensorLangParser::ConfigDirectiveContext>(i);
}

std::vector<TensorLangParser::StatementContext *> TensorLangParser::ProgramContext::statement() {
  return getRuleContexts<TensorLangParser::StatementContext>();
}

TensorLangParser::StatementContext* TensorLangParser::ProgramContext::statement(size_t i) {
  return getRuleContext<TensorLangParser::StatementContext>(i);
}


size_t TensorLangParser::ProgramContext::getRuleIndex() const {
  return TensorLangParser::RuleProgram;
}

void TensorLangParser::ProgramContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TensorLangListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterProgram(this);
}

void TensorLangParser::ProgramContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TensorLangListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitProgram(this);
}


std::any TensorLangParser::ProgramContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<TensorLangVisitor*>(visitor))
    return parserVisitor->visitProgram(this);
  else
    return visitor->visitChildren(this);
}

TensorLangParser::ProgramContext* TensorLangParser::program() {
  ProgramContext *_localctx = _tracker.createInstance<ProgramContext>(_ctx, getState());
  enterRule(_localctx, 0, TensorLangParser::RuleProgram);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(43);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == TensorLangParser::T__0) {
      setState(40);
      configDirective();
      setState(45);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
    setState(47); 
    _errHandler->sync(this);
    _la = _input->LA(1);
    do {
      setState(46);
      statement();
      setState(49); 
      _errHandler->sync(this);
      _la = _input->LA(1);
    } while ((((_la & ~ 0x3fULL) == 0) &&
      ((1ULL << _la) & 18014398509613312) != 0));
    setState(51);
    match(TensorLangParser::EOF);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- ConfigDirectiveContext ------------------------------------------------------------------

TensorLangParser::ConfigDirectiveContext::ConfigDirectiveContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}


size_t TensorLangParser::ConfigDirectiveContext::getRuleIndex() const {
  return TensorLangParser::RuleConfigDirective;
}

void TensorLangParser::ConfigDirectiveContext::copyFrom(ConfigDirectiveContext *ctx) {
  ParserRuleContext::copyFrom(ctx);
}

//----------------- OptimizerConfigContext ------------------------------------------------------------------

tree::TerminalNode* TensorLangParser::OptimizerConfigContext::OPT_TYPE() {
  return getToken(TensorLangParser::OPT_TYPE, 0);
}

TensorLangParser::OptimizerConfigContext::OptimizerConfigContext(ConfigDirectiveContext *ctx) { copyFrom(ctx); }

void TensorLangParser::OptimizerConfigContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TensorLangListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterOptimizerConfig(this);
}
void TensorLangParser::OptimizerConfigContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TensorLangListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitOptimizerConfig(this);
}

std::any TensorLangParser::OptimizerConfigContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<TensorLangVisitor*>(visitor))
    return parserVisitor->visitOptimizerConfig(this);
  else
    return visitor->visitChildren(this);
}
//----------------- DecayConfigContext ------------------------------------------------------------------

tree::TerminalNode* TensorLangParser::DecayConfigContext::DECAY_TYPE() {
  return getToken(TensorLangParser::DECAY_TYPE, 0);
}

tree::TerminalNode* TensorLangParser::DecayConfigContext::FLOAT() {
  return getToken(TensorLangParser::FLOAT, 0);
}

TensorLangParser::DecayConfigContext::DecayConfigContext(ConfigDirectiveContext *ctx) { copyFrom(ctx); }

void TensorLangParser::DecayConfigContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TensorLangListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterDecayConfig(this);
}
void TensorLangParser::DecayConfigContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TensorLangListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitDecayConfig(this);
}

std::any TensorLangParser::DecayConfigContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<TensorLangVisitor*>(visitor))
    return parserVisitor->visitDecayConfig(this);
  else
    return visitor->visitChildren(this);
}
//----------------- DeviceConfigContext ------------------------------------------------------------------

tree::TerminalNode* TensorLangParser::DeviceConfigContext::DEVICE() {
  return getToken(TensorLangParser::DEVICE, 0);
}

TensorLangParser::DeviceConfigContext::DeviceConfigContext(ConfigDirectiveContext *ctx) { copyFrom(ctx); }

void TensorLangParser::DeviceConfigContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TensorLangListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterDeviceConfig(this);
}
void TensorLangParser::DeviceConfigContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TensorLangListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitDeviceConfig(this);
}

std::any TensorLangParser::DeviceConfigContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<TensorLangVisitor*>(visitor))
    return parserVisitor->visitDeviceConfig(this);
  else
    return visitor->visitChildren(this);
}
//----------------- ClusterConfigContext ------------------------------------------------------------------

tree::TerminalNode* TensorLangParser::ClusterConfigContext::BOOLEAN() {
  return getToken(TensorLangParser::BOOLEAN, 0);
}

TensorLangParser::ClusterConfigContext::ClusterConfigContext(ConfigDirectiveContext *ctx) { copyFrom(ctx); }

void TensorLangParser::ClusterConfigContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TensorLangListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterClusterConfig(this);
}
void TensorLangParser::ClusterConfigContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TensorLangListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitClusterConfig(this);
}

std::any TensorLangParser::ClusterConfigContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<TensorLangVisitor*>(visitor))
    return parserVisitor->visitClusterConfig(this);
  else
    return visitor->visitChildren(this);
}
TensorLangParser::ConfigDirectiveContext* TensorLangParser::configDirective() {
  ConfigDirectiveContext *_localctx = _tracker.createInstance<ConfigDirectiveContext>(_ctx, getState());
  enterRule(_localctx, 2, TensorLangParser::RuleConfigDirective);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    setState(70);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 2, _ctx)) {
    case 1: {
      _localctx = _tracker.createInstance<TensorLangParser::DeviceConfigContext>(_localctx);
      enterOuterAlt(_localctx, 1);
      setState(53);
      match(TensorLangParser::T__0);
      setState(54);
      match(TensorLangParser::T__1);
      setState(55);
      match(TensorLangParser::DEVICE);
      setState(56);
      match(TensorLangParser::T__2);
      break;
    }

    case 2: {
      _localctx = _tracker.createInstance<TensorLangParser::ClusterConfigContext>(_localctx);
      enterOuterAlt(_localctx, 2);
      setState(57);
      match(TensorLangParser::T__0);
      setState(58);
      match(TensorLangParser::T__3);
      setState(59);
      match(TensorLangParser::BOOLEAN);
      setState(60);
      match(TensorLangParser::T__2);
      break;
    }

    case 3: {
      _localctx = _tracker.createInstance<TensorLangParser::OptimizerConfigContext>(_localctx);
      enterOuterAlt(_localctx, 3);
      setState(61);
      match(TensorLangParser::T__0);
      setState(62);
      match(TensorLangParser::T__4);
      setState(63);
      match(TensorLangParser::OPT_TYPE);
      setState(64);
      match(TensorLangParser::T__2);
      break;
    }

    case 4: {
      _localctx = _tracker.createInstance<TensorLangParser::DecayConfigContext>(_localctx);
      enterOuterAlt(_localctx, 4);
      setState(65);
      match(TensorLangParser::T__0);
      setState(66);
      match(TensorLangParser::T__5);
      setState(67);
      match(TensorLangParser::DECAY_TYPE);
      setState(68);
      match(TensorLangParser::FLOAT);
      setState(69);
      match(TensorLangParser::T__2);
      break;
    }

    default:
      break;
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- StatementContext ------------------------------------------------------------------

TensorLangParser::StatementContext::StatementContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

TensorLangParser::TensorDeclarationContext* TensorLangParser::StatementContext::tensorDeclaration() {
  return getRuleContext<TensorLangParser::TensorDeclarationContext>(0);
}

TensorLangParser::ActiveEpochBlockContext* TensorLangParser::StatementContext::activeEpochBlock() {
  return getRuleContext<TensorLangParser::ActiveEpochBlockContext>(0);
}

TensorLangParser::AssignmentContext* TensorLangParser::StatementContext::assignment() {
  return getRuleContext<TensorLangParser::AssignmentContext>(0);
}


size_t TensorLangParser::StatementContext::getRuleIndex() const {
  return TensorLangParser::RuleStatement;
}

void TensorLangParser::StatementContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TensorLangListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterStatement(this);
}

void TensorLangParser::StatementContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TensorLangListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitStatement(this);
}


std::any TensorLangParser::StatementContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<TensorLangVisitor*>(visitor))
    return parserVisitor->visitStatement(this);
  else
    return visitor->visitChildren(this);
}

TensorLangParser::StatementContext* TensorLangParser::statement() {
  StatementContext *_localctx = _tracker.createInstance<StatementContext>(_ctx, getState());
  enterRule(_localctx, 4, TensorLangParser::RuleStatement);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    setState(75);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case TensorLangParser::T__7: {
        enterOuterAlt(_localctx, 1);
        setState(72);
        tensorDeclaration();
        break;
      }

      case TensorLangParser::T__16: {
        enterOuterAlt(_localctx, 2);
        setState(73);
        activeEpochBlock();
        break;
      }

      case TensorLangParser::ID: {
        enterOuterAlt(_localctx, 3);
        setState(74);
        assignment();
        break;
      }

    default:
      throw NoViableAltException(this);
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- TensorDeclarationContext ------------------------------------------------------------------

TensorLangParser::TensorDeclarationContext::TensorDeclarationContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

TensorLangParser::TypeIDContext* TensorLangParser::TensorDeclarationContext::typeID() {
  return getRuleContext<TensorLangParser::TypeIDContext>(0);
}

tree::TerminalNode* TensorLangParser::TensorDeclarationContext::ID() {
  return getToken(TensorLangParser::ID, 0);
}

TensorLangParser::ExprContext* TensorLangParser::TensorDeclarationContext::expr() {
  return getRuleContext<TensorLangParser::ExprContext>(0);
}


size_t TensorLangParser::TensorDeclarationContext::getRuleIndex() const {
  return TensorLangParser::RuleTensorDeclaration;
}

void TensorLangParser::TensorDeclarationContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TensorLangListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterTensorDeclaration(this);
}

void TensorLangParser::TensorDeclarationContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TensorLangListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitTensorDeclaration(this);
}


std::any TensorLangParser::TensorDeclarationContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<TensorLangVisitor*>(visitor))
    return parserVisitor->visitTensorDeclaration(this);
  else
    return visitor->visitChildren(this);
}

TensorLangParser::TensorDeclarationContext* TensorLangParser::tensorDeclaration() {
  TensorDeclarationContext *_localctx = _tracker.createInstance<TensorDeclarationContext>(_ctx, getState());
  enterRule(_localctx, 6, TensorLangParser::RuleTensorDeclaration);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(77);
    typeID();
    setState(78);
    match(TensorLangParser::ID);
    setState(79);
    match(TensorLangParser::T__6);
    setState(80);
    expr(0);
    setState(81);
    match(TensorLangParser::T__2);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- TypeIDContext ------------------------------------------------------------------

TensorLangParser::TypeIDContext::TypeIDContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

TensorLangParser::DatatypeContext* TensorLangParser::TypeIDContext::datatype() {
  return getRuleContext<TensorLangParser::DatatypeContext>(0);
}

TensorLangParser::LayoutContext* TensorLangParser::TypeIDContext::layout() {
  return getRuleContext<TensorLangParser::LayoutContext>(0);
}

TensorLangParser::DimSeqContext* TensorLangParser::TypeIDContext::dimSeq() {
  return getRuleContext<TensorLangParser::DimSeqContext>(0);
}


size_t TensorLangParser::TypeIDContext::getRuleIndex() const {
  return TensorLangParser::RuleTypeID;
}

void TensorLangParser::TypeIDContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TensorLangListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterTypeID(this);
}

void TensorLangParser::TypeIDContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TensorLangListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitTypeID(this);
}


std::any TensorLangParser::TypeIDContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<TensorLangVisitor*>(visitor))
    return parserVisitor->visitTypeID(this);
  else
    return visitor->visitChildren(this);
}

TensorLangParser::TypeIDContext* TensorLangParser::typeID() {
  TypeIDContext *_localctx = _tracker.createInstance<TypeIDContext>(_ctx, getState());
  enterRule(_localctx, 8, TensorLangParser::RuleTypeID);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(83);
    match(TensorLangParser::T__7);
    setState(84);
    match(TensorLangParser::T__8);
    setState(85);
    datatype();
    setState(86);
    match(TensorLangParser::T__9);
    setState(87);
    layout();
    setState(88);
    match(TensorLangParser::T__9);
    setState(89);
    dimSeq();
    setState(90);
    match(TensorLangParser::T__10);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- DatatypeContext ------------------------------------------------------------------

TensorLangParser::DatatypeContext::DatatypeContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}


size_t TensorLangParser::DatatypeContext::getRuleIndex() const {
  return TensorLangParser::RuleDatatype;
}

void TensorLangParser::DatatypeContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TensorLangListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterDatatype(this);
}

void TensorLangParser::DatatypeContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TensorLangListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitDatatype(this);
}


std::any TensorLangParser::DatatypeContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<TensorLangVisitor*>(visitor))
    return parserVisitor->visitDatatype(this);
  else
    return visitor->visitChildren(this);
}

TensorLangParser::DatatypeContext* TensorLangParser::datatype() {
  DatatypeContext *_localctx = _tracker.createInstance<DatatypeContext>(_ctx, getState());
  enterRule(_localctx, 10, TensorLangParser::RuleDatatype);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(92);
    _la = _input->LA(1);
    if (!((((_la & ~ 0x3fULL) == 0) &&
      ((1ULL << _la) & 28672) != 0))) {
    _errHandler->recoverInline(this);
    }
    else {
      _errHandler->reportMatch(this);
      consume();
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- LayoutContext ------------------------------------------------------------------

TensorLangParser::LayoutContext::LayoutContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}


size_t TensorLangParser::LayoutContext::getRuleIndex() const {
  return TensorLangParser::RuleLayout;
}

void TensorLangParser::LayoutContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TensorLangListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterLayout(this);
}

void TensorLangParser::LayoutContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TensorLangListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitLayout(this);
}


std::any TensorLangParser::LayoutContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<TensorLangVisitor*>(visitor))
    return parserVisitor->visitLayout(this);
  else
    return visitor->visitChildren(this);
}

TensorLangParser::LayoutContext* TensorLangParser::layout() {
  LayoutContext *_localctx = _tracker.createInstance<LayoutContext>(_ctx, getState());
  enterRule(_localctx, 12, TensorLangParser::RuleLayout);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(94);
    _la = _input->LA(1);
    if (!(_la == TensorLangParser::T__14

    || _la == TensorLangParser::T__15)) {
    _errHandler->recoverInline(this);
    }
    else {
      _errHandler->reportMatch(this);
      consume();
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- DimSeqContext ------------------------------------------------------------------

TensorLangParser::DimSeqContext::DimSeqContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

std::vector<TensorLangParser::DimTokenContext *> TensorLangParser::DimSeqContext::dimToken() {
  return getRuleContexts<TensorLangParser::DimTokenContext>();
}

TensorLangParser::DimTokenContext* TensorLangParser::DimSeqContext::dimToken(size_t i) {
  return getRuleContext<TensorLangParser::DimTokenContext>(i);
}


size_t TensorLangParser::DimSeqContext::getRuleIndex() const {
  return TensorLangParser::RuleDimSeq;
}

void TensorLangParser::DimSeqContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TensorLangListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterDimSeq(this);
}

void TensorLangParser::DimSeqContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TensorLangListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitDimSeq(this);
}


std::any TensorLangParser::DimSeqContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<TensorLangVisitor*>(visitor))
    return parserVisitor->visitDimSeq(this);
  else
    return visitor->visitChildren(this);
}

TensorLangParser::DimSeqContext* TensorLangParser::dimSeq() {
  DimSeqContext *_localctx = _tracker.createInstance<DimSeqContext>(_ctx, getState());
  enterRule(_localctx, 14, TensorLangParser::RuleDimSeq);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(96);
    dimToken();
    setState(101);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == TensorLangParser::T__9) {
      setState(97);
      match(TensorLangParser::T__9);
      setState(98);
      dimToken();
      setState(103);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- DimTokenContext ------------------------------------------------------------------

TensorLangParser::DimTokenContext::DimTokenContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* TensorLangParser::DimTokenContext::INT() {
  return getToken(TensorLangParser::INT, 0);
}

tree::TerminalNode* TensorLangParser::DimTokenContext::ID() {
  return getToken(TensorLangParser::ID, 0);
}


size_t TensorLangParser::DimTokenContext::getRuleIndex() const {
  return TensorLangParser::RuleDimToken;
}

void TensorLangParser::DimTokenContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TensorLangListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterDimToken(this);
}

void TensorLangParser::DimTokenContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TensorLangListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitDimToken(this);
}


std::any TensorLangParser::DimTokenContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<TensorLangVisitor*>(visitor))
    return parserVisitor->visitDimToken(this);
  else
    return visitor->visitChildren(this);
}

TensorLangParser::DimTokenContext* TensorLangParser::dimToken() {
  DimTokenContext *_localctx = _tracker.createInstance<DimTokenContext>(_ctx, getState());
  enterRule(_localctx, 16, TensorLangParser::RuleDimToken);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(104);
    _la = _input->LA(1);
    if (!(_la == TensorLangParser::INT

    || _la == TensorLangParser::ID)) {
    _errHandler->recoverInline(this);
    }
    else {
      _errHandler->reportMatch(this);
      consume();
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- ActiveEpochBlockContext ------------------------------------------------------------------

TensorLangParser::ActiveEpochBlockContext::ActiveEpochBlockContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* TensorLangParser::ActiveEpochBlockContext::INT() {
  return getToken(TensorLangParser::INT, 0);
}

tree::TerminalNode* TensorLangParser::ActiveEpochBlockContext::ID() {
  return getToken(TensorLangParser::ID, 0);
}

tree::TerminalNode* TensorLangParser::ActiveEpochBlockContext::FLOAT() {
  return getToken(TensorLangParser::FLOAT, 0);
}

std::vector<TensorLangParser::StatementContext *> TensorLangParser::ActiveEpochBlockContext::statement() {
  return getRuleContexts<TensorLangParser::StatementContext>();
}

TensorLangParser::StatementContext* TensorLangParser::ActiveEpochBlockContext::statement(size_t i) {
  return getRuleContext<TensorLangParser::StatementContext>(i);
}


size_t TensorLangParser::ActiveEpochBlockContext::getRuleIndex() const {
  return TensorLangParser::RuleActiveEpochBlock;
}

void TensorLangParser::ActiveEpochBlockContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TensorLangListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterActiveEpochBlock(this);
}

void TensorLangParser::ActiveEpochBlockContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TensorLangListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitActiveEpochBlock(this);
}


std::any TensorLangParser::ActiveEpochBlockContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<TensorLangVisitor*>(visitor))
    return parserVisitor->visitActiveEpochBlock(this);
  else
    return visitor->visitChildren(this);
}

TensorLangParser::ActiveEpochBlockContext* TensorLangParser::activeEpochBlock() {
  ActiveEpochBlockContext *_localctx = _tracker.createInstance<ActiveEpochBlockContext>(_ctx, getState());
  enterRule(_localctx, 18, TensorLangParser::RuleActiveEpochBlock);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(106);
    match(TensorLangParser::T__16);
    setState(107);
    match(TensorLangParser::T__17);
    setState(108);
    match(TensorLangParser::INT);
    setState(109);
    match(TensorLangParser::T__9);
    setState(110);
    match(TensorLangParser::ID);
    setState(111);
    match(TensorLangParser::T__9);
    setState(112);
    match(TensorLangParser::FLOAT);
    setState(113);
    match(TensorLangParser::T__18);
    setState(114);
    match(TensorLangParser::T__19);
    setState(116); 
    _errHandler->sync(this);
    _la = _input->LA(1);
    do {
      setState(115);
      statement();
      setState(118); 
      _errHandler->sync(this);
      _la = _input->LA(1);
    } while ((((_la & ~ 0x3fULL) == 0) &&
      ((1ULL << _la) & 18014398509613312) != 0));
    setState(120);
    match(TensorLangParser::T__20);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- AssignmentContext ------------------------------------------------------------------

TensorLangParser::AssignmentContext::AssignmentContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* TensorLangParser::AssignmentContext::ID() {
  return getToken(TensorLangParser::ID, 0);
}

TensorLangParser::ExprContext* TensorLangParser::AssignmentContext::expr() {
  return getRuleContext<TensorLangParser::ExprContext>(0);
}


size_t TensorLangParser::AssignmentContext::getRuleIndex() const {
  return TensorLangParser::RuleAssignment;
}

void TensorLangParser::AssignmentContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TensorLangListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterAssignment(this);
}

void TensorLangParser::AssignmentContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TensorLangListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitAssignment(this);
}


std::any TensorLangParser::AssignmentContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<TensorLangVisitor*>(visitor))
    return parserVisitor->visitAssignment(this);
  else
    return visitor->visitChildren(this);
}

TensorLangParser::AssignmentContext* TensorLangParser::assignment() {
  AssignmentContext *_localctx = _tracker.createInstance<AssignmentContext>(_ctx, getState());
  enterRule(_localctx, 20, TensorLangParser::RuleAssignment);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(122);
    match(TensorLangParser::ID);
    setState(123);
    match(TensorLangParser::T__6);
    setState(124);
    expr(0);
    setState(125);
    match(TensorLangParser::T__2);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- ExprContext ------------------------------------------------------------------

TensorLangParser::ExprContext::ExprContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}


size_t TensorLangParser::ExprContext::getRuleIndex() const {
  return TensorLangParser::RuleExpr;
}

void TensorLangParser::ExprContext::copyFrom(ExprContext *ctx) {
  ParserRuleContext::copyFrom(ctx);
}

//----------------- CrossExprContext ------------------------------------------------------------------

std::vector<TensorLangParser::ExprContext *> TensorLangParser::CrossExprContext::expr() {
  return getRuleContexts<TensorLangParser::ExprContext>();
}

TensorLangParser::ExprContext* TensorLangParser::CrossExprContext::expr(size_t i) {
  return getRuleContext<TensorLangParser::ExprContext>(i);
}

TensorLangParser::CrossExprContext::CrossExprContext(ExprContext *ctx) { copyFrom(ctx); }

void TensorLangParser::CrossExprContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TensorLangListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterCrossExpr(this);
}
void TensorLangParser::CrossExprContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TensorLangListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitCrossExpr(this);
}

std::any TensorLangParser::CrossExprContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<TensorLangVisitor*>(visitor))
    return parserVisitor->visitCrossExpr(this);
  else
    return visitor->visitChildren(this);
}
//----------------- ThetaJoinExprContext ------------------------------------------------------------------

std::vector<TensorLangParser::ExprContext *> TensorLangParser::ThetaJoinExprContext::expr() {
  return getRuleContexts<TensorLangParser::ExprContext>();
}

TensorLangParser::ExprContext* TensorLangParser::ThetaJoinExprContext::expr(size_t i) {
  return getRuleContext<TensorLangParser::ExprContext>(i);
}

TensorLangParser::ThetaJoinExprContext::ThetaJoinExprContext(ExprContext *ctx) { copyFrom(ctx); }

void TensorLangParser::ThetaJoinExprContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TensorLangListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterThetaJoinExpr(this);
}
void TensorLangParser::ThetaJoinExprContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TensorLangListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitThetaJoinExpr(this);
}

std::any TensorLangParser::ThetaJoinExprContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<TensorLangVisitor*>(visitor))
    return parserVisitor->visitThetaJoinExpr(this);
  else
    return visitor->visitChildren(this);
}
//----------------- ExistentialExprContext ------------------------------------------------------------------

TensorLangParser::ExprContext* TensorLangParser::ExistentialExprContext::expr() {
  return getRuleContext<TensorLangParser::ExprContext>(0);
}

TensorLangParser::DimSeqContext* TensorLangParser::ExistentialExprContext::dimSeq() {
  return getRuleContext<TensorLangParser::DimSeqContext>(0);
}

TensorLangParser::LambdaPredContext* TensorLangParser::ExistentialExprContext::lambdaPred() {
  return getRuleContext<TensorLangParser::LambdaPredContext>(0);
}

TensorLangParser::ExistentialExprContext::ExistentialExprContext(ExprContext *ctx) { copyFrom(ctx); }

void TensorLangParser::ExistentialExprContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TensorLangListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterExistentialExpr(this);
}
void TensorLangParser::ExistentialExprContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TensorLangListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitExistentialExpr(this);
}

std::any TensorLangParser::ExistentialExprContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<TensorLangVisitor*>(visitor))
    return parserVisitor->visitExistentialExpr(this);
  else
    return visitor->visitChildren(this);
}
//----------------- IdExprContext ------------------------------------------------------------------

tree::TerminalNode* TensorLangParser::IdExprContext::ID() {
  return getToken(TensorLangParser::ID, 0);
}

TensorLangParser::IdExprContext::IdExprContext(ExprContext *ctx) { copyFrom(ctx); }

void TensorLangParser::IdExprContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TensorLangListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterIdExpr(this);
}
void TensorLangParser::IdExprContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TensorLangListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitIdExpr(this);
}

std::any TensorLangParser::IdExprContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<TensorLangVisitor*>(visitor))
    return parserVisitor->visitIdExpr(this);
  else
    return visitor->visitChildren(this);
}
//----------------- ScalarExprContext ------------------------------------------------------------------

TensorLangParser::TypeIDContext* TensorLangParser::ScalarExprContext::typeID() {
  return getRuleContext<TensorLangParser::TypeIDContext>(0);
}

tree::TerminalNode* TensorLangParser::ScalarExprContext::FLOAT() {
  return getToken(TensorLangParser::FLOAT, 0);
}

TensorLangParser::ScalarExprContext::ScalarExprContext(ExprContext *ctx) { copyFrom(ctx); }

void TensorLangParser::ScalarExprContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TensorLangListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterScalarExpr(this);
}
void TensorLangParser::ScalarExprContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TensorLangListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitScalarExpr(this);
}

std::any TensorLangParser::ScalarExprContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<TensorLangVisitor*>(visitor))
    return parserVisitor->visitScalarExpr(this);
  else
    return visitor->visitChildren(this);
}
//----------------- ReduceAllExprContext ------------------------------------------------------------------

TensorLangParser::ExprContext* TensorLangParser::ReduceAllExprContext::expr() {
  return getRuleContext<TensorLangParser::ExprContext>(0);
}

TensorLangParser::ReduceAllExprContext::ReduceAllExprContext(ExprContext *ctx) { copyFrom(ctx); }

void TensorLangParser::ReduceAllExprContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TensorLangListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterReduceAllExpr(this);
}
void TensorLangParser::ReduceAllExprContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TensorLangListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitReduceAllExpr(this);
}

std::any TensorLangParser::ReduceAllExprContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<TensorLangVisitor*>(visitor))
    return parserVisitor->visitReduceAllExpr(this);
  else
    return visitor->visitChildren(this);
}
//----------------- SubExprContext ------------------------------------------------------------------

std::vector<TensorLangParser::ExprContext *> TensorLangParser::SubExprContext::expr() {
  return getRuleContexts<TensorLangParser::ExprContext>();
}

TensorLangParser::ExprContext* TensorLangParser::SubExprContext::expr(size_t i) {
  return getRuleContext<TensorLangParser::ExprContext>(i);
}

TensorLangParser::SubExprContext::SubExprContext(ExprContext *ctx) { copyFrom(ctx); }

void TensorLangParser::SubExprContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TensorLangListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterSubExpr(this);
}
void TensorLangParser::SubExprContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TensorLangListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitSubExpr(this);
}

std::any TensorLangParser::SubExprContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<TensorLangVisitor*>(visitor))
    return parserVisitor->visitSubExpr(this);
  else
    return visitor->visitChildren(this);
}
//----------------- UniversalExprContext ------------------------------------------------------------------

TensorLangParser::ExprContext* TensorLangParser::UniversalExprContext::expr() {
  return getRuleContext<TensorLangParser::ExprContext>(0);
}

TensorLangParser::DimSeqContext* TensorLangParser::UniversalExprContext::dimSeq() {
  return getRuleContext<TensorLangParser::DimSeqContext>(0);
}

TensorLangParser::LambdaPredContext* TensorLangParser::UniversalExprContext::lambdaPred() {
  return getRuleContext<TensorLangParser::LambdaPredContext>(0);
}

TensorLangParser::UniversalExprContext::UniversalExprContext(ExprContext *ctx) { copyFrom(ctx); }

void TensorLangParser::UniversalExprContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TensorLangListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterUniversalExpr(this);
}
void TensorLangParser::UniversalExprContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TensorLangListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitUniversalExpr(this);
}

std::any TensorLangParser::UniversalExprContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<TensorLangVisitor*>(visitor))
    return parserVisitor->visitUniversalExpr(this);
  else
    return visitor->visitChildren(this);
}
//----------------- AddExprContext ------------------------------------------------------------------

std::vector<TensorLangParser::ExprContext *> TensorLangParser::AddExprContext::expr() {
  return getRuleContexts<TensorLangParser::ExprContext>();
}

TensorLangParser::ExprContext* TensorLangParser::AddExprContext::expr(size_t i) {
  return getRuleContext<TensorLangParser::ExprContext>(i);
}

TensorLangParser::AddExprContext::AddExprContext(ExprContext *ctx) { copyFrom(ctx); }

void TensorLangParser::AddExprContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TensorLangListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterAddExpr(this);
}
void TensorLangParser::AddExprContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TensorLangListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitAddExpr(this);
}

std::any TensorLangParser::AddExprContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<TensorLangVisitor*>(visitor))
    return parserVisitor->visitAddExpr(this);
  else
    return visitor->visitChildren(this);
}
//----------------- UnaryExprContext ------------------------------------------------------------------

TensorLangParser::ExprContext* TensorLangParser::UnaryExprContext::expr() {
  return getRuleContext<TensorLangParser::ExprContext>(0);
}

tree::TerminalNode* TensorLangParser::UnaryExprContext::CELL_OP() {
  return getToken(TensorLangParser::CELL_OP, 0);
}

TensorLangParser::UnaryExprContext::UnaryExprContext(ExprContext *ctx) { copyFrom(ctx); }

void TensorLangParser::UnaryExprContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TensorLangListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterUnaryExpr(this);
}
void TensorLangParser::UnaryExprContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TensorLangListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitUnaryExpr(this);
}

std::any TensorLangParser::UnaryExprContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<TensorLangVisitor*>(visitor))
    return parserVisitor->visitUnaryExpr(this);
  else
    return visitor->visitChildren(this);
}
//----------------- PermuteExprContext ------------------------------------------------------------------

TensorLangParser::ExprContext* TensorLangParser::PermuteExprContext::expr() {
  return getRuleContext<TensorLangParser::ExprContext>(0);
}

TensorLangParser::DimSeqContext* TensorLangParser::PermuteExprContext::dimSeq() {
  return getRuleContext<TensorLangParser::DimSeqContext>(0);
}

TensorLangParser::PermuteExprContext::PermuteExprContext(ExprContext *ctx) { copyFrom(ctx); }

void TensorLangParser::PermuteExprContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TensorLangListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterPermuteExpr(this);
}
void TensorLangParser::PermuteExprContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TensorLangListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitPermuteExpr(this);
}

std::any TensorLangParser::PermuteExprContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<TensorLangVisitor*>(visitor))
    return parserVisitor->visitPermuteExpr(this);
  else
    return visitor->visitChildren(this);
}
//----------------- GatherExprContext ------------------------------------------------------------------

std::vector<TensorLangParser::ExprContext *> TensorLangParser::GatherExprContext::expr() {
  return getRuleContexts<TensorLangParser::ExprContext>();
}

TensorLangParser::ExprContext* TensorLangParser::GatherExprContext::expr(size_t i) {
  return getRuleContext<TensorLangParser::ExprContext>(i);
}

TensorLangParser::GatherExprContext::GatherExprContext(ExprContext *ctx) { copyFrom(ctx); }

void TensorLangParser::GatherExprContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TensorLangListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterGatherExpr(this);
}
void TensorLangParser::GatherExprContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TensorLangListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitGatherExpr(this);
}

std::any TensorLangParser::GatherExprContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<TensorLangVisitor*>(visitor))
    return parserVisitor->visitGatherExpr(this);
  else
    return visitor->visitChildren(this);
}
//----------------- LoadExprContext ------------------------------------------------------------------

std::vector<tree::TerminalNode *> TensorLangParser::LoadExprContext::STRING() {
  return getTokens(TensorLangParser::STRING);
}

tree::TerminalNode* TensorLangParser::LoadExprContext::STRING(size_t i) {
  return getToken(TensorLangParser::STRING, i);
}

TensorLangParser::LoadExprContext::LoadExprContext(ExprContext *ctx) { copyFrom(ctx); }

void TensorLangParser::LoadExprContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TensorLangListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterLoadExpr(this);
}
void TensorLangParser::LoadExprContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TensorLangListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitLoadExpr(this);
}

std::any TensorLangParser::LoadExprContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<TensorLangVisitor*>(visitor))
    return parserVisitor->visitLoadExpr(this);
  else
    return visitor->visitChildren(this);
}
//----------------- HadamardExprContext ------------------------------------------------------------------

std::vector<TensorLangParser::ExprContext *> TensorLangParser::HadamardExprContext::expr() {
  return getRuleContexts<TensorLangParser::ExprContext>();
}

TensorLangParser::ExprContext* TensorLangParser::HadamardExprContext::expr(size_t i) {
  return getRuleContext<TensorLangParser::ExprContext>(i);
}

TensorLangParser::HadamardExprContext::HadamardExprContext(ExprContext *ctx) { copyFrom(ctx); }

void TensorLangParser::HadamardExprContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TensorLangListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterHadamardExpr(this);
}
void TensorLangParser::HadamardExprContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TensorLangListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitHadamardExpr(this);
}

std::any TensorLangParser::HadamardExprContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<TensorLangVisitor*>(visitor))
    return parserVisitor->visitHadamardExpr(this);
  else
    return visitor->visitChildren(this);
}
//----------------- ReduceAxisExprContext ------------------------------------------------------------------

TensorLangParser::ExprContext* TensorLangParser::ReduceAxisExprContext::expr() {
  return getRuleContext<TensorLangParser::ExprContext>(0);
}

tree::TerminalNode* TensorLangParser::ReduceAxisExprContext::INT() {
  return getToken(TensorLangParser::INT, 0);
}

TensorLangParser::ReduceAxisExprContext::ReduceAxisExprContext(ExprContext *ctx) { copyFrom(ctx); }

void TensorLangParser::ReduceAxisExprContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TensorLangListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterReduceAxisExpr(this);
}
void TensorLangParser::ReduceAxisExprContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TensorLangListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitReduceAxisExpr(this);
}

std::any TensorLangParser::ReduceAxisExprContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<TensorLangVisitor*>(visitor))
    return parserVisitor->visitReduceAxisExpr(this);
  else
    return visitor->visitChildren(this);
}
//----------------- MatMulExprContext ------------------------------------------------------------------

std::vector<TensorLangParser::ExprContext *> TensorLangParser::MatMulExprContext::expr() {
  return getRuleContexts<TensorLangParser::ExprContext>();
}

TensorLangParser::ExprContext* TensorLangParser::MatMulExprContext::expr(size_t i) {
  return getRuleContext<TensorLangParser::ExprContext>(i);
}

TensorLangParser::MatMulExprContext::MatMulExprContext(ExprContext *ctx) { copyFrom(ctx); }

void TensorLangParser::MatMulExprContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TensorLangListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterMatMulExpr(this);
}
void TensorLangParser::MatMulExprContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TensorLangListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitMatMulExpr(this);
}

std::any TensorLangParser::MatMulExprContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<TensorLangVisitor*>(visitor))
    return parserVisitor->visitMatMulExpr(this);
  else
    return visitor->visitChildren(this);
}

TensorLangParser::ExprContext* TensorLangParser::expr() {
   return expr(0);
}

TensorLangParser::ExprContext* TensorLangParser::expr(int precedence) {
  ParserRuleContext *parentContext = _ctx;
  size_t parentState = getState();
  TensorLangParser::ExprContext *_localctx = _tracker.createInstance<ExprContext>(_ctx, parentState);
  TensorLangParser::ExprContext *previousContext = _localctx;
  (void)previousContext; // Silence compiler, in case the context is not used by generated code.
  size_t startState = 22;
  enterRecursionRule(_localctx, 22, TensorLangParser::RuleExpr, precedence);

    

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    unrollRecursionContexts(parentContext);
  });
  try {
    size_t alt;
    enterOuterAlt(_localctx, 1);
    setState(143);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case TensorLangParser::ID: {
        _localctx = _tracker.createInstance<IdExprContext>(_localctx);
        _ctx = _localctx;
        previousContext = _localctx;

        setState(128);
        match(TensorLangParser::ID);
        break;
      }

      case TensorLangParser::T__21: {
        _localctx = _tracker.createInstance<LoadExprContext>(_localctx);
        _ctx = _localctx;
        previousContext = _localctx;
        setState(129);
        match(TensorLangParser::T__21);
        setState(130);
        match(TensorLangParser::T__17);
        setState(131);
        match(TensorLangParser::STRING);
        setState(132);
        match(TensorLangParser::T__9);
        setState(133);
        match(TensorLangParser::STRING);
        setState(134);
        match(TensorLangParser::T__18);
        break;
      }

      case TensorLangParser::T__22: {
        _localctx = _tracker.createInstance<ScalarExprContext>(_localctx);
        _ctx = _localctx;
        previousContext = _localctx;
        setState(135);
        match(TensorLangParser::T__22);
        setState(136);
        match(TensorLangParser::T__8);
        setState(137);
        typeID();
        setState(138);
        match(TensorLangParser::T__10);
        setState(139);
        match(TensorLangParser::T__17);
        setState(140);
        match(TensorLangParser::FLOAT);
        setState(141);
        match(TensorLangParser::T__18);
        break;
      }

    default:
      throw NoViableAltException(this);
    }
    _ctx->stop = _input->LT(-1);
    setState(216);
    _errHandler->sync(this);
    alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 8, _ctx);
    while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
      if (alt == 1) {
        if (!_parseListeners.empty())
          triggerExitRuleEvent();
        previousContext = _localctx;
        setState(214);
        _errHandler->sync(this);
        switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 7, _ctx)) {
        case 1: {
          auto newContext = _tracker.createInstance<MatMulExprContext>(_tracker.createInstance<ExprContext>(parentContext, parentState));
          _localctx = newContext;
          pushNewRecursionContext(newContext, startState, RuleExpr);
          setState(145);

          if (!(precpred(_ctx, 13))) throw FailedPredicateException(this, "precpred(_ctx, 13)");
          setState(146);
          match(TensorLangParser::T__23);
          setState(147);
          expr(14);
          break;
        }

        case 2: {
          auto newContext = _tracker.createInstance<AddExprContext>(_tracker.createInstance<ExprContext>(parentContext, parentState));
          _localctx = newContext;
          pushNewRecursionContext(newContext, startState, RuleExpr);
          setState(148);

          if (!(precpred(_ctx, 12))) throw FailedPredicateException(this, "precpred(_ctx, 12)");
          setState(149);
          match(TensorLangParser::T__24);
          setState(150);
          expr(13);
          break;
        }

        case 3: {
          auto newContext = _tracker.createInstance<SubExprContext>(_tracker.createInstance<ExprContext>(parentContext, parentState));
          _localctx = newContext;
          pushNewRecursionContext(newContext, startState, RuleExpr);
          setState(151);

          if (!(precpred(_ctx, 11))) throw FailedPredicateException(this, "precpred(_ctx, 11)");
          setState(152);
          match(TensorLangParser::T__25);
          setState(153);
          expr(12);
          break;
        }

        case 4: {
          auto newContext = _tracker.createInstance<CrossExprContext>(_tracker.createInstance<ExprContext>(parentContext, parentState));
          _localctx = newContext;
          pushNewRecursionContext(newContext, startState, RuleExpr);
          setState(154);

          if (!(precpred(_ctx, 10))) throw FailedPredicateException(this, "precpred(_ctx, 10)");
          setState(155);
          match(TensorLangParser::T__26);
          setState(156);
          expr(11);
          break;
        }

        case 5: {
          auto newContext = _tracker.createInstance<HadamardExprContext>(_tracker.createInstance<ExprContext>(parentContext, parentState));
          _localctx = newContext;
          pushNewRecursionContext(newContext, startState, RuleExpr);
          setState(157);

          if (!(precpred(_ctx, 9))) throw FailedPredicateException(this, "precpred(_ctx, 9)");
          setState(158);
          match(TensorLangParser::T__27);
          setState(159);
          match(TensorLangParser::T__17);
          setState(160);
          expr(0);
          setState(161);
          match(TensorLangParser::T__18);
          break;
        }

        case 6: {
          auto newContext = _tracker.createInstance<UnaryExprContext>(_tracker.createInstance<ExprContext>(parentContext, parentState));
          _localctx = newContext;
          pushNewRecursionContext(newContext, startState, RuleExpr);
          setState(163);

          if (!(precpred(_ctx, 8))) throw FailedPredicateException(this, "precpred(_ctx, 8)");
          setState(164);
          match(TensorLangParser::T__28);
          setState(165);
          match(TensorLangParser::T__8);
          setState(166);
          match(TensorLangParser::CELL_OP);
          setState(167);
          match(TensorLangParser::T__10);
          setState(168);
          match(TensorLangParser::T__29);
          break;
        }

        case 7: {
          auto newContext = _tracker.createInstance<ReduceAllExprContext>(_tracker.createInstance<ExprContext>(parentContext, parentState));
          _localctx = newContext;
          pushNewRecursionContext(newContext, startState, RuleExpr);
          setState(169);

          if (!(precpred(_ctx, 7))) throw FailedPredicateException(this, "precpred(_ctx, 7)");
          setState(170);
          match(TensorLangParser::T__30);
          break;
        }

        case 8: {
          auto newContext = _tracker.createInstance<ReduceAxisExprContext>(_tracker.createInstance<ExprContext>(parentContext, parentState));
          _localctx = newContext;
          pushNewRecursionContext(newContext, startState, RuleExpr);
          setState(171);

          if (!(precpred(_ctx, 6))) throw FailedPredicateException(this, "precpred(_ctx, 6)");
          setState(172);
          match(TensorLangParser::T__31);
          setState(173);
          match(TensorLangParser::T__8);
          setState(174);
          match(TensorLangParser::INT);
          setState(175);
          match(TensorLangParser::T__10);
          setState(176);
          match(TensorLangParser::T__29);
          break;
        }

        case 9: {
          auto newContext = _tracker.createInstance<PermuteExprContext>(_tracker.createInstance<ExprContext>(parentContext, parentState));
          _localctx = newContext;
          pushNewRecursionContext(newContext, startState, RuleExpr);
          setState(177);

          if (!(precpred(_ctx, 5))) throw FailedPredicateException(this, "precpred(_ctx, 5)");
          setState(178);
          match(TensorLangParser::T__32);
          setState(179);
          match(TensorLangParser::T__8);
          setState(180);
          dimSeq();
          setState(181);
          match(TensorLangParser::T__10);
          setState(182);
          match(TensorLangParser::T__29);
          break;
        }

        case 10: {
          auto newContext = _tracker.createInstance<GatherExprContext>(_tracker.createInstance<ExprContext>(parentContext, parentState));
          _localctx = newContext;
          pushNewRecursionContext(newContext, startState, RuleExpr);
          setState(184);

          if (!(precpred(_ctx, 4))) throw FailedPredicateException(this, "precpred(_ctx, 4)");
          setState(185);
          match(TensorLangParser::T__33);
          setState(186);
          match(TensorLangParser::T__17);
          setState(187);
          expr(0);
          setState(188);
          match(TensorLangParser::T__18);
          break;
        }

        case 11: {
          auto newContext = _tracker.createInstance<ThetaJoinExprContext>(_tracker.createInstance<ExprContext>(parentContext, parentState));
          _localctx = newContext;
          pushNewRecursionContext(newContext, startState, RuleExpr);
          setState(190);

          if (!(precpred(_ctx, 3))) throw FailedPredicateException(this, "precpred(_ctx, 3)");
          setState(191);
          match(TensorLangParser::T__34);
          setState(192);
          match(TensorLangParser::T__17);
          setState(193);
          expr(0);
          setState(194);
          match(TensorLangParser::T__18);
          break;
        }

        case 12: {
          auto newContext = _tracker.createInstance<ExistentialExprContext>(_tracker.createInstance<ExprContext>(parentContext, parentState));
          _localctx = newContext;
          pushNewRecursionContext(newContext, startState, RuleExpr);
          setState(196);

          if (!(precpred(_ctx, 2))) throw FailedPredicateException(this, "precpred(_ctx, 2)");
          setState(197);
          match(TensorLangParser::T__35);
          setState(198);
          match(TensorLangParser::T__8);
          setState(199);
          dimSeq();
          setState(200);
          match(TensorLangParser::T__10);
          setState(201);
          match(TensorLangParser::T__17);
          setState(202);
          lambdaPred();
          setState(203);
          match(TensorLangParser::T__18);
          break;
        }

        case 13: {
          auto newContext = _tracker.createInstance<UniversalExprContext>(_tracker.createInstance<ExprContext>(parentContext, parentState));
          _localctx = newContext;
          pushNewRecursionContext(newContext, startState, RuleExpr);
          setState(205);

          if (!(precpred(_ctx, 1))) throw FailedPredicateException(this, "precpred(_ctx, 1)");
          setState(206);
          match(TensorLangParser::T__36);
          setState(207);
          match(TensorLangParser::T__8);
          setState(208);
          dimSeq();
          setState(209);
          match(TensorLangParser::T__10);
          setState(210);
          match(TensorLangParser::T__17);
          setState(211);
          lambdaPred();
          setState(212);
          match(TensorLangParser::T__18);
          break;
        }

        default:
          break;
        } 
      }
      setState(218);
      _errHandler->sync(this);
      alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 8, _ctx);
    }
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }
  return _localctx;
}

//----------------- LambdaPredContext ------------------------------------------------------------------

TensorLangParser::LambdaPredContext::LambdaPredContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* TensorLangParser::LambdaPredContext::ID() {
  return getToken(TensorLangParser::ID, 0);
}

TensorLangParser::LogicalExprContext* TensorLangParser::LambdaPredContext::logicalExpr() {
  return getRuleContext<TensorLangParser::LogicalExprContext>(0);
}


size_t TensorLangParser::LambdaPredContext::getRuleIndex() const {
  return TensorLangParser::RuleLambdaPred;
}

void TensorLangParser::LambdaPredContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TensorLangListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterLambdaPred(this);
}

void TensorLangParser::LambdaPredContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TensorLangListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitLambdaPred(this);
}


std::any TensorLangParser::LambdaPredContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<TensorLangVisitor*>(visitor))
    return parserVisitor->visitLambdaPred(this);
  else
    return visitor->visitChildren(this);
}

TensorLangParser::LambdaPredContext* TensorLangParser::lambdaPred() {
  LambdaPredContext *_localctx = _tracker.createInstance<LambdaPredContext>(_ctx, getState());
  enterRule(_localctx, 24, TensorLangParser::RuleLambdaPred);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(219);
    match(TensorLangParser::T__17);
    setState(220);
    match(TensorLangParser::ID);
    setState(221);
    match(TensorLangParser::T__18);
    setState(222);
    match(TensorLangParser::T__37);
    setState(223);
    logicalExpr();
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- LogicalExprContext ------------------------------------------------------------------

TensorLangParser::LogicalExprContext::LogicalExprContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

std::vector<TensorLangParser::LogicalAndExprContext *> TensorLangParser::LogicalExprContext::logicalAndExpr() {
  return getRuleContexts<TensorLangParser::LogicalAndExprContext>();
}

TensorLangParser::LogicalAndExprContext* TensorLangParser::LogicalExprContext::logicalAndExpr(size_t i) {
  return getRuleContext<TensorLangParser::LogicalAndExprContext>(i);
}


size_t TensorLangParser::LogicalExprContext::getRuleIndex() const {
  return TensorLangParser::RuleLogicalExpr;
}

void TensorLangParser::LogicalExprContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TensorLangListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterLogicalExpr(this);
}

void TensorLangParser::LogicalExprContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TensorLangListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitLogicalExpr(this);
}


std::any TensorLangParser::LogicalExprContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<TensorLangVisitor*>(visitor))
    return parserVisitor->visitLogicalExpr(this);
  else
    return visitor->visitChildren(this);
}

TensorLangParser::LogicalExprContext* TensorLangParser::logicalExpr() {
  LogicalExprContext *_localctx = _tracker.createInstance<LogicalExprContext>(_ctx, getState());
  enterRule(_localctx, 26, TensorLangParser::RuleLogicalExpr);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(225);
    logicalAndExpr();
    setState(230);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == TensorLangParser::T__38) {
      setState(226);
      match(TensorLangParser::T__38);
      setState(227);
      logicalAndExpr();
      setState(232);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- LogicalAndExprContext ------------------------------------------------------------------

TensorLangParser::LogicalAndExprContext::LogicalAndExprContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

std::vector<TensorLangParser::LogicalNotExprContext *> TensorLangParser::LogicalAndExprContext::logicalNotExpr() {
  return getRuleContexts<TensorLangParser::LogicalNotExprContext>();
}

TensorLangParser::LogicalNotExprContext* TensorLangParser::LogicalAndExprContext::logicalNotExpr(size_t i) {
  return getRuleContext<TensorLangParser::LogicalNotExprContext>(i);
}


size_t TensorLangParser::LogicalAndExprContext::getRuleIndex() const {
  return TensorLangParser::RuleLogicalAndExpr;
}

void TensorLangParser::LogicalAndExprContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TensorLangListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterLogicalAndExpr(this);
}

void TensorLangParser::LogicalAndExprContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TensorLangListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitLogicalAndExpr(this);
}


std::any TensorLangParser::LogicalAndExprContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<TensorLangVisitor*>(visitor))
    return parserVisitor->visitLogicalAndExpr(this);
  else
    return visitor->visitChildren(this);
}

TensorLangParser::LogicalAndExprContext* TensorLangParser::logicalAndExpr() {
  LogicalAndExprContext *_localctx = _tracker.createInstance<LogicalAndExprContext>(_ctx, getState());
  enterRule(_localctx, 28, TensorLangParser::RuleLogicalAndExpr);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(233);
    logicalNotExpr();
    setState(238);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == TensorLangParser::T__39) {
      setState(234);
      match(TensorLangParser::T__39);
      setState(235);
      logicalNotExpr();
      setState(240);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- LogicalNotExprContext ------------------------------------------------------------------

TensorLangParser::LogicalNotExprContext::LogicalNotExprContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}


size_t TensorLangParser::LogicalNotExprContext::getRuleIndex() const {
  return TensorLangParser::RuleLogicalNotExpr;
}

void TensorLangParser::LogicalNotExprContext::copyFrom(LogicalNotExprContext *ctx) {
  ParserRuleContext::copyFrom(ctx);
}

//----------------- IntrinsicPredicateContext ------------------------------------------------------------------

tree::TerminalNode* TensorLangParser::IntrinsicPredicateContext::INTRINSIC_OP() {
  return getToken(TensorLangParser::INTRINSIC_OP, 0);
}

TensorLangParser::IntrinsicPredicateContext::IntrinsicPredicateContext(LogicalNotExprContext *ctx) { copyFrom(ctx); }

void TensorLangParser::IntrinsicPredicateContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TensorLangListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterIntrinsicPredicate(this);
}
void TensorLangParser::IntrinsicPredicateContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TensorLangListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitIntrinsicPredicate(this);
}

std::any TensorLangParser::IntrinsicPredicateContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<TensorLangVisitor*>(visitor))
    return parserVisitor->visitIntrinsicPredicate(this);
  else
    return visitor->visitChildren(this);
}
//----------------- SubPredicateContext ------------------------------------------------------------------

TensorLangParser::LogicalExprContext* TensorLangParser::SubPredicateContext::logicalExpr() {
  return getRuleContext<TensorLangParser::LogicalExprContext>(0);
}

TensorLangParser::SubPredicateContext::SubPredicateContext(LogicalNotExprContext *ctx) { copyFrom(ctx); }

void TensorLangParser::SubPredicateContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TensorLangListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterSubPredicate(this);
}
void TensorLangParser::SubPredicateContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TensorLangListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitSubPredicate(this);
}

std::any TensorLangParser::SubPredicateContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<TensorLangVisitor*>(visitor))
    return parserVisitor->visitSubPredicate(this);
  else
    return visitor->visitChildren(this);
}
//----------------- CompPredicateContext ------------------------------------------------------------------

TensorLangParser::FloatComparisonContext* TensorLangParser::CompPredicateContext::floatComparison() {
  return getRuleContext<TensorLangParser::FloatComparisonContext>(0);
}

TensorLangParser::CompPredicateContext::CompPredicateContext(LogicalNotExprContext *ctx) { copyFrom(ctx); }

void TensorLangParser::CompPredicateContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TensorLangListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterCompPredicate(this);
}
void TensorLangParser::CompPredicateContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TensorLangListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitCompPredicate(this);
}

std::any TensorLangParser::CompPredicateContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<TensorLangVisitor*>(visitor))
    return parserVisitor->visitCompPredicate(this);
  else
    return visitor->visitChildren(this);
}
//----------------- NotPredicateContext ------------------------------------------------------------------

TensorLangParser::LogicalNotExprContext* TensorLangParser::NotPredicateContext::logicalNotExpr() {
  return getRuleContext<TensorLangParser::LogicalNotExprContext>(0);
}

TensorLangParser::NotPredicateContext::NotPredicateContext(LogicalNotExprContext *ctx) { copyFrom(ctx); }

void TensorLangParser::NotPredicateContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TensorLangListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterNotPredicate(this);
}
void TensorLangParser::NotPredicateContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TensorLangListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitNotPredicate(this);
}

std::any TensorLangParser::NotPredicateContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<TensorLangVisitor*>(visitor))
    return parserVisitor->visitNotPredicate(this);
  else
    return visitor->visitChildren(this);
}
TensorLangParser::LogicalNotExprContext* TensorLangParser::logicalNotExpr() {
  LogicalNotExprContext *_localctx = _tracker.createInstance<LogicalNotExprContext>(_ctx, getState());
  enterRule(_localctx, 30, TensorLangParser::RuleLogicalNotExpr);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    setState(249);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case TensorLangParser::T__40: {
        _localctx = _tracker.createInstance<TensorLangParser::NotPredicateContext>(_localctx);
        enterOuterAlt(_localctx, 1);
        setState(241);
        match(TensorLangParser::T__40);
        setState(242);
        logicalNotExpr();
        break;
      }

      case TensorLangParser::T__17: {
        _localctx = _tracker.createInstance<TensorLangParser::SubPredicateContext>(_localctx);
        enterOuterAlt(_localctx, 2);
        setState(243);
        match(TensorLangParser::T__17);
        setState(244);
        logicalExpr();
        setState(245);
        match(TensorLangParser::T__18);
        break;
      }

      case TensorLangParser::T__25:
      case TensorLangParser::T__42:
      case TensorLangParser::T__43:
      case TensorLangParser::FLOAT:
      case TensorLangParser::ID: {
        _localctx = _tracker.createInstance<TensorLangParser::CompPredicateContext>(_localctx);
        enterOuterAlt(_localctx, 3);
        setState(247);
        floatComparison();
        break;
      }

      case TensorLangParser::INTRINSIC_OP: {
        _localctx = _tracker.createInstance<TensorLangParser::IntrinsicPredicateContext>(_localctx);
        enterOuterAlt(_localctx, 4);
        setState(248);
        match(TensorLangParser::INTRINSIC_OP);
        break;
      }

    default:
      throw NoViableAltException(this);
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- FloatComparisonContext ------------------------------------------------------------------

TensorLangParser::FloatComparisonContext::FloatComparisonContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

std::vector<TensorLangParser::FloatExprContext *> TensorLangParser::FloatComparisonContext::floatExpr() {
  return getRuleContexts<TensorLangParser::FloatExprContext>();
}

TensorLangParser::FloatExprContext* TensorLangParser::FloatComparisonContext::floatExpr(size_t i) {
  return getRuleContext<TensorLangParser::FloatExprContext>(i);
}

tree::TerminalNode* TensorLangParser::FloatComparisonContext::COMP_OP() {
  return getToken(TensorLangParser::COMP_OP, 0);
}


size_t TensorLangParser::FloatComparisonContext::getRuleIndex() const {
  return TensorLangParser::RuleFloatComparison;
}

void TensorLangParser::FloatComparisonContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TensorLangListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterFloatComparison(this);
}

void TensorLangParser::FloatComparisonContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TensorLangListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitFloatComparison(this);
}


std::any TensorLangParser::FloatComparisonContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<TensorLangVisitor*>(visitor))
    return parserVisitor->visitFloatComparison(this);
  else
    return visitor->visitChildren(this);
}

TensorLangParser::FloatComparisonContext* TensorLangParser::floatComparison() {
  FloatComparisonContext *_localctx = _tracker.createInstance<FloatComparisonContext>(_ctx, getState());
  enterRule(_localctx, 32, TensorLangParser::RuleFloatComparison);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(251);
    floatExpr();
    setState(252);
    match(TensorLangParser::COMP_OP);
    setState(253);
    floatExpr();
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- FloatExprContext ------------------------------------------------------------------

TensorLangParser::FloatExprContext::FloatExprContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

std::vector<TensorLangParser::FloatTermContext *> TensorLangParser::FloatExprContext::floatTerm() {
  return getRuleContexts<TensorLangParser::FloatTermContext>();
}

TensorLangParser::FloatTermContext* TensorLangParser::FloatExprContext::floatTerm(size_t i) {
  return getRuleContext<TensorLangParser::FloatTermContext>(i);
}


size_t TensorLangParser::FloatExprContext::getRuleIndex() const {
  return TensorLangParser::RuleFloatExpr;
}

void TensorLangParser::FloatExprContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TensorLangListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterFloatExpr(this);
}

void TensorLangParser::FloatExprContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TensorLangListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitFloatExpr(this);
}


std::any TensorLangParser::FloatExprContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<TensorLangVisitor*>(visitor))
    return parserVisitor->visitFloatExpr(this);
  else
    return visitor->visitChildren(this);
}

TensorLangParser::FloatExprContext* TensorLangParser::floatExpr() {
  FloatExprContext *_localctx = _tracker.createInstance<FloatExprContext>(_ctx, getState());
  enterRule(_localctx, 34, TensorLangParser::RuleFloatExpr);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(255);
    floatTerm();
    setState(260);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == TensorLangParser::T__24

    || _la == TensorLangParser::T__25) {
      setState(256);
      _la = _input->LA(1);
      if (!(_la == TensorLangParser::T__24

      || _la == TensorLangParser::T__25)) {
      _errHandler->recoverInline(this);
      }
      else {
        _errHandler->reportMatch(this);
        consume();
      }
      setState(257);
      floatTerm();
      setState(262);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- FloatTermContext ------------------------------------------------------------------

TensorLangParser::FloatTermContext::FloatTermContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

std::vector<TensorLangParser::FloatFactorContext *> TensorLangParser::FloatTermContext::floatFactor() {
  return getRuleContexts<TensorLangParser::FloatFactorContext>();
}

TensorLangParser::FloatFactorContext* TensorLangParser::FloatTermContext::floatFactor(size_t i) {
  return getRuleContext<TensorLangParser::FloatFactorContext>(i);
}


size_t TensorLangParser::FloatTermContext::getRuleIndex() const {
  return TensorLangParser::RuleFloatTerm;
}

void TensorLangParser::FloatTermContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TensorLangListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterFloatTerm(this);
}

void TensorLangParser::FloatTermContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TensorLangListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitFloatTerm(this);
}


std::any TensorLangParser::FloatTermContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<TensorLangVisitor*>(visitor))
    return parserVisitor->visitFloatTerm(this);
  else
    return visitor->visitChildren(this);
}

TensorLangParser::FloatTermContext* TensorLangParser::floatTerm() {
  FloatTermContext *_localctx = _tracker.createInstance<FloatTermContext>(_ctx, getState());
  enterRule(_localctx, 36, TensorLangParser::RuleFloatTerm);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(263);
    floatFactor();
    setState(268);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == TensorLangParser::T__23

    || _la == TensorLangParser::T__41) {
      setState(264);
      _la = _input->LA(1);
      if (!(_la == TensorLangParser::T__23

      || _la == TensorLangParser::T__41)) {
      _errHandler->recoverInline(this);
      }
      else {
        _errHandler->reportMatch(this);
        consume();
      }
      setState(265);
      floatFactor();
      setState(270);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- FloatFactorContext ------------------------------------------------------------------

TensorLangParser::FloatFactorContext::FloatFactorContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}


size_t TensorLangParser::FloatFactorContext::getRuleIndex() const {
  return TensorLangParser::RuleFloatFactor;
}

void TensorLangParser::FloatFactorContext::copyFrom(FloatFactorContext *ctx) {
  ParserRuleContext::copyFrom(ctx);
}

//----------------- NegateFloatExprContext ------------------------------------------------------------------

TensorLangParser::FloatFactorContext* TensorLangParser::NegateFloatExprContext::floatFactor() {
  return getRuleContext<TensorLangParser::FloatFactorContext>(0);
}

TensorLangParser::NegateFloatExprContext::NegateFloatExprContext(FloatFactorContext *ctx) { copyFrom(ctx); }

void TensorLangParser::NegateFloatExprContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TensorLangListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterNegateFloatExpr(this);
}
void TensorLangParser::NegateFloatExprContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TensorLangListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitNegateFloatExpr(this);
}

std::any TensorLangParser::NegateFloatExprContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<TensorLangVisitor*>(visitor))
    return parserVisitor->visitNegateFloatExpr(this);
  else
    return visitor->visitChildren(this);
}
//----------------- CellFloatExprContext ------------------------------------------------------------------

TensorLangParser::CellFloatExprContext::CellFloatExprContext(FloatFactorContext *ctx) { copyFrom(ctx); }

void TensorLangParser::CellFloatExprContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TensorLangListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterCellFloatExpr(this);
}
void TensorLangParser::CellFloatExprContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TensorLangListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitCellFloatExpr(this);
}

std::any TensorLangParser::CellFloatExprContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<TensorLangVisitor*>(visitor))
    return parserVisitor->visitCellFloatExpr(this);
  else
    return visitor->visitChildren(this);
}
//----------------- AbsFloatExprContext ------------------------------------------------------------------

TensorLangParser::FloatExprContext* TensorLangParser::AbsFloatExprContext::floatExpr() {
  return getRuleContext<TensorLangParser::FloatExprContext>(0);
}

TensorLangParser::AbsFloatExprContext::AbsFloatExprContext(FloatFactorContext *ctx) { copyFrom(ctx); }

void TensorLangParser::AbsFloatExprContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TensorLangListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterAbsFloatExpr(this);
}
void TensorLangParser::AbsFloatExprContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TensorLangListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitAbsFloatExpr(this);
}

std::any TensorLangParser::AbsFloatExprContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<TensorLangVisitor*>(visitor))
    return parserVisitor->visitAbsFloatExpr(this);
  else
    return visitor->visitChildren(this);
}
//----------------- IdFloatExprContext ------------------------------------------------------------------

tree::TerminalNode* TensorLangParser::IdFloatExprContext::ID() {
  return getToken(TensorLangParser::ID, 0);
}

TensorLangParser::IdFloatExprContext::IdFloatExprContext(FloatFactorContext *ctx) { copyFrom(ctx); }

void TensorLangParser::IdFloatExprContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TensorLangListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterIdFloatExpr(this);
}
void TensorLangParser::IdFloatExprContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TensorLangListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitIdFloatExpr(this);
}

std::any TensorLangParser::IdFloatExprContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<TensorLangVisitor*>(visitor))
    return parserVisitor->visitIdFloatExpr(this);
  else
    return visitor->visitChildren(this);
}
//----------------- LiteralFloatExprContext ------------------------------------------------------------------

tree::TerminalNode* TensorLangParser::LiteralFloatExprContext::FLOAT() {
  return getToken(TensorLangParser::FLOAT, 0);
}

TensorLangParser::LiteralFloatExprContext::LiteralFloatExprContext(FloatFactorContext *ctx) { copyFrom(ctx); }

void TensorLangParser::LiteralFloatExprContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TensorLangListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterLiteralFloatExpr(this);
}
void TensorLangParser::LiteralFloatExprContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TensorLangListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitLiteralFloatExpr(this);
}

std::any TensorLangParser::LiteralFloatExprContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<TensorLangVisitor*>(visitor))
    return parserVisitor->visitLiteralFloatExpr(this);
  else
    return visitor->visitChildren(this);
}
TensorLangParser::FloatFactorContext* TensorLangParser::floatFactor() {
  FloatFactorContext *_localctx = _tracker.createInstance<FloatFactorContext>(_ctx, getState());
  enterRule(_localctx, 38, TensorLangParser::RuleFloatFactor);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    setState(281);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case TensorLangParser::T__42: {
        _localctx = _tracker.createInstance<TensorLangParser::AbsFloatExprContext>(_localctx);
        enterOuterAlt(_localctx, 1);
        setState(271);
        match(TensorLangParser::T__42);
        setState(272);
        match(TensorLangParser::T__17);
        setState(273);
        floatExpr();
        setState(274);
        match(TensorLangParser::T__18);
        break;
      }

      case TensorLangParser::T__43: {
        _localctx = _tracker.createInstance<TensorLangParser::CellFloatExprContext>(_localctx);
        enterOuterAlt(_localctx, 2);
        setState(276);
        match(TensorLangParser::T__43);
        break;
      }

      case TensorLangParser::ID: {
        _localctx = _tracker.createInstance<TensorLangParser::IdFloatExprContext>(_localctx);
        enterOuterAlt(_localctx, 3);
        setState(277);
        match(TensorLangParser::ID);
        break;
      }

      case TensorLangParser::FLOAT: {
        _localctx = _tracker.createInstance<TensorLangParser::LiteralFloatExprContext>(_localctx);
        enterOuterAlt(_localctx, 4);
        setState(278);
        match(TensorLangParser::FLOAT);
        break;
      }

      case TensorLangParser::T__25: {
        _localctx = _tracker.createInstance<TensorLangParser::NegateFloatExprContext>(_localctx);
        enterOuterAlt(_localctx, 5);
        setState(279);
        match(TensorLangParser::T__25);
        setState(280);
        floatFactor();
        break;
      }

    default:
      throw NoViableAltException(this);
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

bool TensorLangParser::sempred(RuleContext *context, size_t ruleIndex, size_t predicateIndex) {
  switch (ruleIndex) {
    case 11: return exprSempred(antlrcpp::downCast<ExprContext *>(context), predicateIndex);

  default:
    break;
  }
  return true;
}

bool TensorLangParser::exprSempred(ExprContext *_localctx, size_t predicateIndex) {
  switch (predicateIndex) {
    case 0: return precpred(_ctx, 13);
    case 1: return precpred(_ctx, 12);
    case 2: return precpred(_ctx, 11);
    case 3: return precpred(_ctx, 10);
    case 4: return precpred(_ctx, 9);
    case 5: return precpred(_ctx, 8);
    case 6: return precpred(_ctx, 7);
    case 7: return precpred(_ctx, 6);
    case 8: return precpred(_ctx, 5);
    case 9: return precpred(_ctx, 4);
    case 10: return precpred(_ctx, 3);
    case 11: return precpred(_ctx, 2);
    case 12: return precpred(_ctx, 1);

  default:
    break;
  }
  return true;
}

void TensorLangParser::initialize() {
#if ANTLR4_USE_THREAD_LOCAL_CACHE
  tensorlangParserInitialize();
#else
  ::antlr4::internal::call_once(tensorlangParserOnceFlag, tensorlangParserInitialize);
#endif
}
