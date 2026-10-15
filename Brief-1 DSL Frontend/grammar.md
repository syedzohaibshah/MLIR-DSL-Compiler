# TLDR Grammar Reference

Formal grammar for the `.tldr` kernel language.

---

## Program

```
program      ::= func_def+
func_def     ::= name '(' param_list ')' ['->' '[' ident_list ']' ']' '{' block '}'
param_list   ::= (param (',' param)*)?
param        ::= (IDENT ':' type_anno)?
ident_list   ::= IDENT (',' IDENT)*
```

---

## Types

```
type_anno    ::= 'Tensor' '[' dtype ',' '(' shape ')' ']'
               | dtype
dtype        ::= 'f32' | 'f64' | 'i32'
shape        ::= shape_dim (',' shape_dim)*
shape_dim    ::= INTEGER | IDENT
```

---

## Statements

```
block        ::= stmt*
stmt         ::= for_stmt
               | if_stmt
               | return_stmt
               | print_stmt
               | 'break' [';']
               | 'continue' [';']
               | assign_or_decl

for_stmt     ::= 'for' IDENT 'in' 'loop' '(' expr (',' IDENT '=' INTEGER)* ')' '{' block '}'
if_stmt      ::= 'if' expr '{' block '}' ['else' '{' block '}']
return_stmt  ::= 'return' [expr] [';']
print_stmt   ::= 'print' '(' expr ')' [';']

assign_or_decl ::= IDENT ':' type_anno ['=' expr] [';']
               | target assign_op expr [';']
target       ::= IDENT
               | IDENT '[' expr_list ']'
assign_op    ::= '=' | '+=' | '-=' | '*=' | '/='
```

---

## Expressions (precedence lowest to highest)

```
expr         ::= or_expr
or_expr      ::= and_expr ('or' and_expr)*
and_expr     ::= comp_expr ('and' comp_expr)*
comp_expr    ::= add_expr (comp_op add_expr)?
add_expr     ::= mul_expr (('+' | '-') mul_expr)*
mul_expr     ::= unary_expr (('*' | '/') unary_expr)*
unary_expr   ::= '-' unary_expr | primary

primary      ::= INTEGER | FLOAT
               | '(' expr ')'
               | '[' expr_list ']'
               | IDENT '[' expr_list ']'           -- subscript
               | IDENT '(' expr_list ')'           -- function call
               | IDENT
```

### Operator Precedence

| Level | Operators | Associativity |
|-------|-----------|---------------|
| 1 | `or` | left |
| 2 | `and` | left |
| 3 | `<` `>` `<=` `>=` `==` `!=` | left |
| 4 | `+` `-` | left |
| 5 | `*` `/` | left |
| 6 | `-` (unary) | right |

---

## Built-in Operations

```
builtin_call ::= name '(' arg_list ')'
arg_list     ::= (arg (',' arg)*)?
arg          ::= expr
```

`arg ::= expr` means every argument is a **full expression**, so function calls, literals, variables, subscripts, arithmetic, etc. are all valid arguments — including **nested function calls**.

| Name | Arity | Description |
|------|-------|-------------|
| `relu` | 1 | Rectified linear unit |
| `softmax` | 2 | Softmax over input |
| `embedding` | 2 | Embedding lookup |
| `layernorm` | 4 | Layer normalization |
| `mha_attention` | 3 | Multi-head attention (GQA) |

### Valid nesting examples

```
softmax(relu(A), x)                     # relu result passed to softmax
print(softmax(A, B))                    # builtin nested in print
layernorm(softmax(A, B), C, D, E)       # deeply nested
embedding(A, relu(B))                   # relu result passed to embedding
```

### Invalid (not supported)

```
apply(softmax, A)                       # function reference not allowed
softmax                                 # bare keyword without call
```
