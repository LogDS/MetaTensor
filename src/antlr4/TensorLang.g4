grammar TensorLang;

program         : configDirective* statement+ EOF ;

configDirective : 'use' 'device' DEVICE ';'       # DeviceConfig
                | 'use' 'cluster' BOOLEAN ';'     # ClusterConfig
                | 'use' 'optimizer' OPT_TYPE ';'  # OptimizerConfig
                | 'use' 'decay' DECAY_TYPE FLOAT ';' # DecayConfig
                ;

statement       : tensorDeclaration
                | activeEpochBlock
                | assignment
                ;

tensorDeclaration : typeID ID '=' expr ';' ;

typeID          : 'MetaTensor' '<' datatype ',' layout ',' dimSeq '>' ;
datatype        : 'float' | 'double' | 'int' ;
layout          : 'Dense' | 'SparseCOO' ;

// Supporto duale simbolico e numerico
dimSeq          : dimToken (',' dimToken)* ;
dimToken        : INT | ID ;

activeEpochBlock : 'epoch_loop' '(' INT ',' ID ',' FLOAT ')' '{' statement+ '}' ;

assignment      : ID '=' expr ';' ;

expr            : ID                                                # IdExpr
                | 'load_safetensors' '(' STRING ',' STRING ')'      # LoadExpr
                | 'from_scalar' '<' typeID '>' '(' FLOAT ')'        # ScalarExpr
                | expr '*' expr                                     # MatMulExpr
                | expr '+' expr                                     # AddExpr
                | expr '-' expr                                     # SubExpr
                | expr '%' expr                                     # CrossExpr
                | expr '.element_wise_mul' '(' expr ')'             # HadamardExpr
                | expr '.apply' '<' CELL_OP '>' '()'                # UnaryExpr
                | expr '.reduce_all_sum()'                          # ReduceAllExpr
                | expr '.reduce_sum' '<' INT '>' '()'               # ReduceAxisExpr
                | expr '.permute_axes' '<' dimSeq '>' '()'          # PermuteExpr
                | expr '.gather_nd' '(' expr ')'                    # GatherExpr
                | expr '.tensor_theta_join' '(' expr ')'            # ThetaJoinExpr
                | expr '.evaluate_existential' '<' dimSeq '>' '(' lambdaPred ')' # ExistentialExpr
                | expr '.evaluate_universal' '<' dimSeq '>' '(' lambdaPred ')'   # UniversalExpr
                ;

// --- GRAMMATICA ESTESA DELLE ESPRESSIONI LAMBDA E PREDICATI LOGICI ---
lambdaPred      : '(' ID ')' '->' logicalExpr ;

logicalExpr     : logicalAndExpr ( '||' logicalAndExpr )* ;
logicalAndExpr  : logicalNotExpr ( '&&' logicalNotExpr )* ;
logicalNotExpr  : '!' logicalNotExpr # NotPredicate
                | '(' logicalExpr ')' # SubPredicate
                | floatComparison    # CompPredicate
                | INTRINSIC_OP       # IntrinsicPredicate
                ;

floatComparison : floatExpr COMP_OP floatExpr ;

floatExpr       : floatTerm (( '+' | '-' ) floatTerm)* ;
floatTerm       : floatFactor (( '*' | '/' ) floatFactor)* ;
floatFactor     : 'abs' '(' floatExpr ')' # AbsFloatExpr
                | 'cell'                  # CellFloatExpr
                | ID                      # IdFloatExpr
                | FLOAT                   # LiteralFloatExpr
                | '-' floatFactor         # NegateFloatExpr
                ;

DEVICE          : 'cpu' | 'cuda' ;
BOOLEAN         : 'true' | 'false' ;
OPT_TYPE        : 'SGD' | 'Momentum' | 'Adam' ;
DECAY_TYPE      : 'None' | 'Step' | 'Exponential' ;
CELL_OP         : 'Sigmoid' | 'Logit' | 'Exp' | 'Log' | 'Tanh' | 'Abs' | 'Sqrt' | 'Square' ;
COMP_OP         : '>' | '<' | '==' | '!=' | '>=' | '<=' ;
INTRINSIC_OP    : 'isnan' | 'eps' | 'plus_infinity' | 'minus_infinity' ;
INT             : [0-9]+ ;
FLOAT           : [0-9]+ '.' [0-9]+ 'f'? ;
ID              : [a-zA-Z_][a-zA-Z0-9_]* ;
STRING          : '"' (~["\\] | '\\' .)* '"' ;
WS              : [ \t\r\n]+ -> skip ;
