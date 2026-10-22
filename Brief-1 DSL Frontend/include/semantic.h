#ifndef TLDR_SEMANTIC_H
#define TLDR_SEMANTIC_H

#include "ast_nodes.h"
#include "builtin_ops.h"
#include <string>
#include <vector>
#include <unordered_map>
#include <iostream>
#include <stdexcept>

class SemanticError : public std::runtime_error {
public:
    explicit SemanticError(const std::string& msg) : std::runtime_error(msg) {}
};

// Symbol table for variables and parameters of a single function scope.
struct TypeInfo {
    bool is_tensor = false;
    bool is_param = false;
    bool initialized = false;
    std::string dtype;
    std::vector<ShapeDim> shape;
};

// Symbol table for functions (global scope).
struct FuncInfo {
    std::string name;
    size_t param_count = 0;
    std::vector<std::string> returns;
    bool is_main = false;
    bool has_explicit_return = false;
    bool returns_tensor = false;
};

class SemanticAnalyzer {
public:
    void analyze(const std::vector<std::shared_ptr<FuncNode>>& funcs);

private:
    std::unordered_map<std::string, TypeInfo> symbol_table_; //hashmap for symbols
    std::unordered_map<std::string, FuncInfo> function_table_; //hashmap for functions

    std::string current_func_;

    // Hard semantic errors gathered during the current analysis; if any are
    // present when analysis completes, analyze() throws a SemanticError.
    std::vector<std::string> errors_;

    // Record a recoverable hard error and continue analysis so all problems
    // can be reported together.
    void semantic_error(const std::string& msg);

    void analyze_func(const std::shared_ptr<FuncNode>& func);
    void analyze_stmt(const std::shared_ptr<ASTNode>& stmt);
    void analyze_expr(const std::shared_ptr<ASTNode>& expr);
    void resolve_var(const std::string& name, int line, int col);
    void resolve_call(const std::string& name, size_t argc, int line, int col);
    bool is_tensor_expr(const std::shared_ptr<ASTNode>& expr);
    bool tensor_is_empty(const std::shared_ptr<ASTNode>& expr);
    bool is_uninitialized_tensor(const std::shared_ptr<ASTNode>& expr);
    void mark_initialized(const std::shared_ptr<ASTNode>& target);
};

#endif
