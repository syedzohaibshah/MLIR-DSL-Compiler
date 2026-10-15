# DSL Syntax Reference

Complete grammar for the `.tldr` kernel language.

---

## 1. Program Structure

A `.tldr` file is a sequence of function definitions. One function may be named `main`; it takes no parameters and is the kernel entry point.

```
program  ::= func_def+
func_def ::= name '(' param_list ')' ['->' '[' ident_list ']' ']' '{' block '}'
```

If present, `main`:
- must **not** take parameters.
- **should** contain an explicit `return` statement.
- may call other user-defined functions declared at the top level.

---

## 2. Function Definition

```
func_def  ::= IDENT '(' param_list ')' ['->' '[' ident_list ']'']' '{' block '}'

param_list  ::= (param (',' param)*)?
param       ::= IDENT ':' type_anno
ident_list  ::= IDENT (',' IDENT)*
```

### Example

```
matmul(
    A : Tensor[f32, (M, K)],
    B : Tensor[f32, (K, N)],
    C : Tensor[f32, (M, N)]
) -> [C] {
    ...
}
```

- **Function name** -- becomes the kernel identifier.
- **Parameters** -- typed arguments (tensors or scalars).
- **Return annotation** -- `-> [name, ...]` lists output variable names. Optional; if absent, `returns` is empty.

### main Function

```
main()
{
    A : Tensor[f32, (64, 64)] = [[0]*64]*64
    B : Tensor[f32, (64, 64)] = [[0]*64]*64
    C : Tensor[f32, (64, 64)] = [[0]*64]*64
    C = matmul(A, B, C)
    print(softmax(A, B))
    return C
}
```

- `main` declares local tensors, calls user-defined functions (e.g., `matmul`), prints results with `print(...)`, and returns a value with `return`.
- The semantic analyzer (see [Architecture](Architecture.md)) warns if `main` takes parameters or lacks an explicit `return`.

---

## 3. Types

### Tensor Types

```
type_anno  ::= 'Tensor' '[' dtype ',' '(' shape ')' ']'
             | dtype

dtype  ::= 'f32' | 'f64' | 'i32'

shape     ::= shape_dim (',' shape_dim)*
shape_dim ::= INTEGER | IDENT
```

Examples:
```
A : Tensor[f32, (M, K)]        # 2-D, symbolic dims
buf : Tensor[i32, (64, 64)]    # 2-D, fixed 64x64
vec : Tensor[f64, (N,)]        # 1-D vector
```

### Scalar Types

```
epsilon : f32
count : i32
```

Supported scalar types: `f32`, `f64`, `i32`.

---

## 4. Statement CFG

Statements form the body of functions, loops, and conditionals.

```
stmt  ::= for_stmt
         | if_stmt
         | return_stmt
         | print_stmt
         | 'break' [';']
         | 'continue' [';']
         | assign_or_decl

block ::= stmt*
```

### print_stmt

```
print_stmt  ::= 'print' '(' expr ')' [';']
```

The `print` statement marks an output-producing expression (used in `main`):

```
print(softmax(A, B))
print(C[i, j])
```

### for_stmt

```
for_stmt  ::= 'for' IDENT 'in' 'loop' '(' expr (',' IDENT '=' INTEGER)* ')' '{' block '}'
```

The `loop()` call takes a bound expression followed by optional `tile=N` and/or `unroll=N` keyword arguments. Exactly one of `tile` or `unroll` may be specified per loop (the parser accepts any combination but the compiler assumes one).

```
for i in loop(M, tile=64) { ... }
for j in loop(N, tile=64, unroll=4) { ... }
for k in loop(64, unroll=2) { ... }
```

### if_stmt

```
if_stmt  ::= 'if' expr '{' block '}' ['else' '{' block '}']
```

```
if A[i] > 0 {
    B[i] = A[i]
} else {
    B[i] = 0
}
```

### return_stmt

```
return_stmt  ::= 'return' [expr] [';']
```

`return` is intended for `main` (the semantic analyzer warns if used in other functions and if `main` lacks a `return`):

```
return C
```

### assign_or_decl

```
assign_or_decl  ::= IDENT ':' type_anno ['=' expr] [';']     # tensor/variable declaration
                  | target assign_op expr [';']                # assignment

target  ::= IDENT                       # simple variable
          | IDENT '[' expr_list ']'     # subscript

assign_op  ::= '=' | '+=' | '-=' | '*=' | '/='
```

Tensor declaration:
```
C_local : Tensor[f32, (64, 64)] = [[0]*64]*64;
sum_val : Tensor[f32, (64,)] = [0]*64;
```

Assignment (simple and augmented):
```
x = expr
C[i, j] = A[i, k] * B[k, j]
C[i, j] += A[i, k] * B[k, j]
sum_val[i] += C_local[i, jj]
```

---

## 5. Expression CFG

Expressions use standard precedence from lowest to highest binding.

```
expr      ::= or_expr
or_expr   ::= and_expr ('or' and_expr)*
and_expr  ::= comp_expr ('and' comp_expr)*
comp_expr ::= add_expr (comp_op add_expr)?
add_expr  ::= mul_expr (('+' | '-') mul_expr)*
mul_expr  ::= unary_expr (('*' | '/') unary_expr)*
unary_expr ::= '-' unary_expr | primary
primary   ::= INTEGER | FLOAT
            | '(' expr ')'
            | '[' expr_list ']'
            | IDENT '[' expr_list ']'
            | IDENT '(' expr_list ')'        # built-in or user function call
            | IDENT
```

### Operator Precedence (lowest to highest)

| Level | Operators | Associativity |
|-------|-----------|---------------|
| 1 | `or` | left |
| 2 | `and` | left |
| 3 | `<` `>` `<=` `>=` `==` `!=` | left (one comparison per expression) |
| 4 | `+` `-` | left |
| 5 | `*` `/` | left |
| 6 | `-` (unary) | right |

### Primary Expressions

**Integer / Float literals:**
```
0
64
3.14
```

**Parenthesized expression:**
```
(C_local[i, j] - mean[i]) * gamma[j] + beta[j]
```

**Subscript (tensor indexing):**
```
A[i, k]            # multi-index
C_local[i, jj]     # multi-index with loop variable
C_local[i]         # single-index
C[i+ii, k]         # index as addition expression
bias[j+jj]         # index as addition expression
```

**Built-in function call:**
```
relu(A)
softmax(A, x)
embedding(A, B)
layernorm(A, B, C, D)
mha_attention(A, B, C)
```

**User-defined function call:**
```
C = matmul(A, B, C)      # any function declared at the top level
print(softmax(A, B))     # built-ins may be nested in print(...)
```

**Variable reference:**
```
M
i
epsilon
```

---

## 6. Built-in Operations

The following built-in operations are recognized as function calls:

| Name | Arity | Description |
|------|-------|-------------|
| `relu` | 1 | Rectified linear unit |
| `softmax` | 2 | Softmax over input |
| `embedding` | 2 | Embedding lookup |
| `layernorm` | 4 | Layer normalization |
| `mha_attention` | 3 | Multi-head attention (GQA) |

---

## 7. Full Example -- Fused GEMM + Bias + LayerNorm

```
fused_gemm_bias_layernorm(
    A : Tensor[f32, (M, K)],
    B : Tensor[f32, (K, N)],
    bias : Tensor[f32, (N,)],
    gamma : Tensor[f32, (N,)],
    beta : Tensor[f32, (N,)],
    epsilon : f32,
    threshold : f32
) -> [C] {
    C_local : Tensor[f32, (64, 64)] = [[0]*64]*64;
    sum_val : Tensor[f32, (64,)] = [0]*64;

    for i in loop(M, tile=64) {
        for j in loop(N, tile=64) {
            for k in loop(K, tile=32, unroll=4) {
                for ii in loop(64, unroll=4) {
                    for jj in loop(64, unroll=2) {
                        C_local[ii, jj] += A[i+ii, k] * B[k, j+jj]
                    }
                }
            }
        }
    }

    for i in loop(M, tile=64) {
        for j in loop(N, tile=64) {
            for jj in loop(64, unroll=8) {
                C_local[i, jj] = C_local[i, jj] + bias[j+jj]
                sum_val[i] += C_local[i, jj]
            }
        }
    }

    for i in loop(M, tile=64) {
        for j in loop(N, tile=64, unroll=4) {
            for jj in loop(64, unroll=2) {
                C[i, jj] = (C_local[i, jj] - mean[i]) * gamma[j+jj] + beta[j+jj]
                C[i, jj] = C[i, jj] * threshold
            }
        }
    }
}
```
