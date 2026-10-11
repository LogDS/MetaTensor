grammar MetaTensor;

program: MODULE ID decl+;

arg_typecast : BOOL             #boolT
         | FLOAT32              #floatT
         | FLOAT64              #doubleT
         | INT32                #int32T
         | INT64                #int64T
         | UINT32               #uint32
         ;
 dimarg : ID                                 #id_dimarg
        | INTEGER                            #int_dimarg
        ;
 dimT   : (DIM '^' dimarg '×')* DIM '^' dimarg 'as' DIM '=' arg_typecast;
 funarg : ID ':' (arg_typecast | dimT);


decl : DIM ID                                                                                   #dim_decl
     | 'fun' ID '(' funarg+ ')' '->' ((arg_typecast | dimT) ',')* (arg_typecast | dimT) ':'
                             statement*
                             'return' ID+                                                       #fun_decl
     ;


lambda_arg : ID                         #lambda_id
           | FLOAT                      #lambda_float
           | lambda_arg '>' lambda_arg  #lambda_gt
           | lambda_arg '<' lambda_arg  #lambda_lt
           | lambda_arg LEQ lambda_arg  #lambda_leq
           | lambda_arg GEQ lambda_arg  #lambda_geq
           | lambda_arg '=' lambda_arg  #lambda_eq
           | lambda_arg NEQ lambda_arg  #lambda_neq
           | lambda_arg L_AND lambda_arg #lambda_log_and
           | lambda_arg L_OR  lambda_arg #lambda_log_or
           | lambda_arg XOR   lambda_arg #lambda_xor
           | NEG lambda_arg              #lambda_neg
           ;

booltensor : ID                                        #booltensor_var
           | INDICATOR lambda_arg                       #bool_tensor_from_indicator
           | EXISTS (INTEGER ',')* INTEGER '.' booltensor   #bool_tensor_exists_proj
           | ALL    (INTEGER ',')* INTEGER '.' booltensor   #bool_tensor_forall_proj
           ;

bool:  ID        #boolvar
    | TRUE              #tt
    | FALSE             #ff
    | EXISTS booltensor #any
    | ALL    booltensor #all
    ;

float: FLOAT                         #float_number
     | ID       #float_var
     | MAX   tensor                  #cellmax
     | MIN   tensor                  #cellmin
     | SUM   tensor                  #cellsum
     | PROD  tensor                  #cellprod
     | float '+.' float              #floatsum
     | float '-.' float              #floatminus
     | float '*.' float              #floattimes
     | float '/.' float              #floatdiv
     | '(|' float '|)'                 #floatpar
     ;

tensor: ID       #tensor_id
      | '(' tensor ')'                 #tensorpar
      | tensor TENSORPROD tensor       #tensorprod
      | tensor HADAMARD   tensor       #cellwise_hadamard
      | tensor '+'        tensor       #tensorsum
      | tensor '-'        tensor       #tensordiff
      | tensor '+'        float       #tensorfloatsum
      | tensor '-'        float       #tensorfloatdiff
      | tensor '*'        float       #tensorproddiff
      | tensor '/'        float       #tensordivdiff
      | PERMUTATION (INTEGER ',')* INTEGER ':' tensor #permute
      | SUM         (INTEGER ',')* INTEGER ':' tensor #sumaggr
      | PROD (INTEGER ',')* INTEGER ':' tensor        #prodaggr
      | '[' booltensor ']_{' float '}^{' float '}'        #clip_over_floats
      | booltensor '?' tensor ':' tensor              #clip_over_precise_values
      | GATHER_ND_OF_INDICES_PREFIX indices=tensor '}(' actual=tensor ')'                      #gathernd
      | STACK tensor tensor+   #stack
      | RESHAPE tensor '}(' tensor ')'      #reshape
      ;


statement: LET ID (':' dimT)? ASSIGN tensor                                 #lettensor
         | LET ID (':' dimT)? ASSIGN booltensor                             #letlambda
         | LET ID (':' (FLOAT32|FLOAT64))? ASSIGN float                     #letfloat
         | WHILE bool DO statement+ DONE                                    #whileloop
         | IF bool THEN statement+ ELSE statement+ ENDIF                    #ifstatement
         ;

RESHAPE: 'Φ_{' | '\\Phi_{';
STACK: '⨁' | '\\oplus';
GATHER_ND_OF_INDICES_PREFIX: '𝜄_{' | '\\iota';
GATHER_ND_OF_INDICES_INFIX: '⊂' | '\\subset';
IF: 'if';
ENDIF: 'endif';
THEN: 'then';
ELSE: 'else';
MAX: 'max';
MIN: 'min';
MODULE: 'module';
LET:'let';
WHILE: 'while';
DO: 'do';
DONE: 'done';
ASSIGN: ':=';
PERMUTATION: 'ℼ' | '\\mathbb{\\pi}';
TENSORPROD : '\\otimes' | '⊗';
HADAMARD:   '\\circ' | '∘';
SUM: '∑' | '\\sum' | 'sum';
PROD: '∏' | '\\prod' | 'prod';
TRUE: 'True' | 'true' | 'tt';
FALSE: 'False' | 'false' | 'ff';
UINT32 : 'ℕ32' | 'uint32' | '\\mathbb{N}32';
INT64 :  'ℤ64' | 'int64' | '\\mathbb{Z}64';
INT32 :  'ℤ32' | 'int32' | '\\mathbb{Z}32';
FLOAT64: '𝔽64' | 'double' | '\\mathbb{F}64';
FLOAT32: '𝔽32' | 'float' | '\\mathbb{F}32';
BOOL: '𝔹' | 'bool' | '\\mathbb{B}';
DIM: 'dim';
L_AND: 'and' | '∧' | '&&';
L_OR: 'or' | '∨' | '||';
NEG: 'not' | '¬' | '!';
XOR: 'xor' | '⊻' | '^';
LEQ: '≤' | '<=';
GEQ: '≥' | '>=';
NEQ: '≠' | '!=' | '<>';
EXISTS: '∃' | 'some' | 'any' | 'exists';
ALL: '∀' | 'all';
INDICATOR: '𝕀' | '\\mathbb{I}';
INTEGER : [-]? DIGIT;
FLOAT : [-]? DecimalFloatingConstant ;
ID              : [a-zA-Z_][a-zA-Z0-9_]* ;
STRING          : '"' (~["\\] | '\\' .)* '"' ;
WS              : [ \t\r\n]+ -> skip ;

WIDE_COMMENT
    : '/*' .*? '*/' -> skip
;

LINE_COMMENT
    : '#' ~[\r\n]* -> skip
;

fragment
DecimalFloatingConstant
    :   [0-9]* '.' DIGIT
        |   DIGIT '.'

    |   DIGIT ExponentPart
    ;

fragment
FractionalConstant
    :   [0-9]* '.' DIGIT
    |   DIGIT '.'
    ;

fragment
ExponentPart
    :   [eE] [-]? DIGIT
    ;

fragment DIGIT : [0-9]+;
