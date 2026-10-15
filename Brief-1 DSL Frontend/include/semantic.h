#ifndef TLDR_SEMANTIC_H
#define TLDR_SEMANTIC_H

#include "ast_nodes.h"
#include "builtin_ops.h"
#include <string>
#include <vector>
#include <unordered_map>
#include <iostream>

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
    // Analyze a set of parsed functions; prints diagnostics to std::cerr.
    void analyze(const std::vector<std::shared_ptr<FuncNode>>& funcs);

private:
    // Two symbol tables (the hashmaps requested):
    // 1. variables & parameters for the currently-analyzed function
    // 2. all declared functions (global)
    std::unordered_map<std::string, TypeInfo> symbol_table_;
    std::unordered_map<std::string, FuncInfo> function_table_;

    std::string current_func_;

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
