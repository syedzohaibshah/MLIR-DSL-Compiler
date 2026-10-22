# Architecture

The repository contains two compiler pipelines for the TLDR DSL.

---

## Pipeline 1 -- Standalone C++ Compiler (`test_tldr_compiler/`)

This is the primary pipeline. A `.tldr` source file goes through lexing, parsing, and visitor-based code generation.

```
.tldr source file
        |
        v
    Lexer (lexer.h / lexer.cpp)
        |
        v
    Token stream (tokens.h)
        |
        v
    Parser (parser.h / parser.cpp)
        |
        v
    AST  (ast_nodes.h / ast_nodes.cpp)
        |
        v
SemanticAnalyzer (semantic.h / semantic.cpp)
    |     * symbol table: variables + parameters
    |     * symbol table: functions
    |     * checks the data of a tensor against its dims
    |
    +---> JsonVisitor        -->  .json file
    +---> GraphvizVisitor    -->  .dot file  (render with `dot`)
```

### Module Descriptions

| Module | File | Role |
|--------|------|------|
| **Tokens** | `include/tokens.h`, `src/tokens.cpp` | Defines `TokenType` enum and `Token` struct. Maps token types to printable names. |
| **Lexer** | `include/lexer.h`, `src/lexer.cpp` | Scans source text into a `vector<Token>`. Handles whitespace, comments, numbers, identifiers, operators, and keywords. |
| **AST Nodes** | `include/ast_nodes.h`, `src/ast_nodes.cpp` | Defines the node hierarchy: `FuncNode`, `ParamNode`, `ForLoopNode`, `IfNode`, `AssignNode`, `TensorDeclNode`, `BinaryOpNode`, `UnaryOpNode`, `SubscriptNode`, `BoolOpNode`, `TernaryNode`, `VariableNode`, `ConstantNode`, `ReturnNode`, `BreakNode`, `ContinueNode`, `ListNode`, `BuiltinCallNode`, `PrintNode`, `FunctionCallNode`. Each node implements `print()` and `accept(Visitor&)`. |
| **Parser** | `include/parser.h`, `src/parser.cpp` | Recursive-descent parser. Consumes tokens and builds the AST. Implements the full grammar: function defs, `main`, blocks, statements (incl. `print`), and expressions with correct precedence. |
| **SemanticAnalyzer** | `include/semantic.h`, `src/semantic.cpp` | Semantic/name-resolution pass. Maintains two symbol tables (hashmaps): one mapping **variables & parameters** to their types per function scope, and one mapping **function names** to their `FuncInfo` (arity, returns, is_main). Emits warnings for undeclared names, undeclared functions, arity mismatches, `main` with parameters, `return` outside `main`, and `main` without an explicit `return`. Checks the data of a tensor declaration against its declared dims (rank, element count, ragged lists) when the init is a constant literal; symbolic dims are resolved from in-scope constants or from call-site bindings. |
| **ConstEval** | `include/const_eval.h`, `src/const_eval.cpp` | Flattens nested list literals of constants, folds simple arithmetic on constants, reconstructs nested JSON for `data_values`. |
| **JsonVisitor** | `include/json_visitor.h`, `src/json_visitor.cpp` | Visitor that serializes the AST to a JSON object using nlohmann/json. Outputs one `.json` file per function. Adds `data_values` (the reconstructed nested array of the tensor initialiser) alongside the existing `data` AST dump. |
| **GraphvizVisitor** | `include/graphviz_visitor.h` | Visitor that emits a Graphviz DOT graph of the AST. Node colors indicate type (orange for loops, yellow for control flow, cyan for params, green for variables, etc.). |
| **Main** | `src/main.cpp` | CLI entry point. Reads `.tldr` file, runs lexer, parser, and semantic analysis; prints AST to stdout, and optionally writes `.json` and `.dot` files. |

### CLI Usage

```
tldr_compiler <file.tldr> [--tokens] [--json] [--dot]
```

### Building

```bash
cd test_tldr_compiler
mkdir build && cd build
cmake ..
cmake --build .
```

Requires: C++17, CMake 3.14+.

---

## Pipeline 2 -- Python Decorator Pipeline (root)

An alternative path that captures Python function source code and converts it to JSON via Python's `ast` module.

```
Python @kernel function
        |
        v
kernel_decorator.py          # inspect.getsource() + ast.parse()
        |
        v
JSON AST payload              # serialized Python AST dict
        |
        v
compiler_backend.pyd          # pybind11 module (calls C++ Parser + Visitor)
        |
        v
Parser.cpp                    # builds ExprNode AST from JSON
        |
        +---> Visitor          # reconstructs JSON from ExprNode tree
        +---> GraphvizASTVisitor   # generates kernel_ast.dot / kernel_ast.png
```

### Module Descriptions

| Module | File | Role |
|--------|------|------|
| **kernel_decorator** | `kernel_decorator.py` | `@kernel` decorator. Captures source via `inspect.getsource()`, parses with `ast.parse()`, serializes AST to JSON dict. |
| **compiler_backend** | `compiler_backend.cpp`, `compiler_backend.pyd` | pybind11 binding. Accepts JSON string, calls C++ Parser to build `ExprNode` tree, runs visitors. |
| **ExprNode** | `Test folder/ExprNode.h`, `Test folder/ExprNode.cpp` | AST node types for the Python-pipeline variant. |
| **Parser** | `Test folder/Parser.h`, `Test folder/Parser.cpp` | Builds `ExprNode` AST from the JSON payload. |

### Output Files

- `kernel_ast.dot` -- Graphviz DOT representation of the full AST.
- `kernel_ast.png` -- Rendered PNG image of the AST (requires `dot` on PATH).

### Building the pybind11 Extension

```bash
g++ -O2 -shared -std=c++17 \
    -I<python_include> -I<pybind11_include> \
    compiler_backend.cpp "Test folder/ExprNode.cpp" "Test folder/Parser.cpp" \
    -o compiler_backend.pyd -L<python_lib> -lpython39
```

---

## Comparison

| Aspect | Pipeline 1 (`test_tldr_compiler`) | Pipeline 2 (Python decorator) |
|--------|-------------------------------------|-------------------------------|
| Input format | `.tldr` text files | Python `@kernel` functions |
| Parser | C++ recursive-descent | Python `ast` + C++ JSON parser |
| AST nodes | 20 node kinds (`ast_nodes.h`) | `ExprNode` variants |
| Outputs | JSON, DOT | JSON, DOT, PNG |
| Status | Active, primary | Earlier prototype |
