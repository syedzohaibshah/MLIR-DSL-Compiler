# TLDR Compiler -- Starter Guide

A standalone C++ compiler for the **TLDR** domain-specific language. TLDR lets you write tensor kernel specifications in a concise, readable syntax. The compiler parses `.tldr` files into an AST, and can export the result as JSON or a Graphviz DOT graph.

## Quick Start

### 1. Build the compiler

```bash
cd test_tldr_compiler
mkdir build && cd build
cmake ..
cmake --build .
```

Requires C++17 and CMake 3.14+.

### 2. Write a kernel

Create a file `matmul.tldr`:

```
matmul(
    A : Tensor[f32, (M, K)],
    B : Tensor[f32, (K, N)],
    C : Tensor[f32, (M, N)]
) -> [C] {
    for i in loop(M, tile=64) {
        for j in loop(N, tile=64) {
            for k in loop(K, tile=32, unroll=4) {
                if i < M and j < N {
                    C[i, j] += A[i, k] * B[k, j]
                }
            }
        }
    }
}
```

### 3. Compile

```bash
# Print the AST to stdout
./tldr_compiler matmul.tldr

# Export JSON AST
./tldr_compiler matmul.tldr --json

# Export Graphviz DOT
./tldr_compiler matmul.tldr --dot

# Show token stream
./tldr_compiler matmul.tldr --tokens
```

### 4. Output

```
matmul.json    # JSON representation of the AST
matmul.dot     # Graphviz DOT file (render with `dot -Tpng matmul.dot -o matmul.png`)
```

---

## Language at a Glance

A `.tldr` file contains one or more **function definitions**. Each function has a name, typed parameters, an optional return list, and a brace-delimited body of statements.

```
kernel_name(
    param : Tensor[f32, (M, N)],
    scalar : f32
) -> [output_tensor] {
    // body
}
```

**Statements** include `for` loops, `if`/`else`, `return`, `print`, `break`, `continue`, assignments, and tensor declarations. **Expressions** cover arithmetic, comparisons, boolean logic, subscripts, built-in calls, and calls to user-defined functions.

### main + user functions

`.tldr` files may declare helpers at the top level and a `main()` entry point that ties them together:

```
matmul(
    A : Tensor[f32, (64, 64)],
    B : Tensor[f32, (64, 64)],
    C : Tensor[f32, (64, 64)]
) -> [C] {
    for i in loop(64, tile=64) {
        C[i, 0] = A[i, 0] * B[0, i]
    }
}

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

- `main` takes **no parameters** and **should return** a value.
- User functions declared outside `main` are callable from inside it.
- `print(...)` marks output-producing expressions.
- The compiler runs a semantic pass using two symbol tables (variables/parameters + functions) to validate names and arity.

See [DSL Syntax Reference](DSL_Syntax.md) for the full grammar.

---

## More Examples

### ReLU with if/else

```
relu(
    A : Tensor[f32, (N,)],
    B : Tensor[f32, (N,)]
) {
    for i in loop(N, tile=128, unroll=4) {
        if A[i] > 0 {
            B[i] = A[i]
        } else {
            B[i] = 0
        }
    }
}
```

### Softmax

```
softmax(
    A : Tensor[f32, (M, N)],
    B : Tensor[f32, (M, N)],
    max_val : Tensor[f32, (M,)],
    sum_exp : Tensor[f32, (M,)]
) {
    for i in loop(M, tile=64) {
        for j in loop(N, tile=64, unroll=4) {
            if A[i, j] > max_val[i] {
                max_val[i] = A[i, j]
            }
        }
    }

    for i in loop(M, tile=64) {
        sum_exp[i] = 0;
        for j in loop(N, tile=64, unroll=4) {
            sum_exp[i] += A[i, j] - max_val[i]
        }
    }

    for i in loop(M, tile=64) {
        for j in loop(N, tile=64, unroll=4) {
            B[i, j] = A[i, j] - max_val[i] - sum_exp[i]
        }
    }
}
```

### Fused GEMM + Bias + LayerNorm

See `test_tldr_compiler/tests/test_fused.tldr` for a full fused kernel with local tensor declarations, tiled matmul, bias addition, and layer normalization -- all in a single function.

---

## Documentation

| Document | Description |
|----------|-------------|
| [DSL Syntax Reference](DSL_Syntax.md) | Full grammar with statement and expression CFGs, data types, loops, assignments, and expressions. |
| [Architecture](Architecture.md) | Compiler pipeline, module descriptions, and build instructions. |
| [Parser Reference](Parser_Reference.md) | Supported and unsupported constructs in the `.tldr` parser. |
| [Resources](Resources.md) | External links for MLIR, LLVM, and related tools. |
