#ifndef TLDR_SEMANTIC_H
#define TLDR_SEMANTIC_H

#include "ast_nodes.h"
#include "builtin_ops.h"
#include "const_eval.h"
#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

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
    // Value of a name that holds a compile-time constant, such as a dimension
    // that was given a number by an assignment.
    std::optional<int64_t> const_value;
};

// Symbol table for functions (global scope).
struct FuncInfo {
    std::string name;
    size_t param_count = 0;
    std::vector<std::string> returns;
    bool is_main = false;
    bool has_explicit_return = false;
    bool returns_tensor = false;
    // Shapes of the parameters, used to bind the symbolic dimensions of a
    // callee to the dimensions of the arguments of its call sites.
    std::vector<std::vector<ShapeDim>> param_shapes;
};

class SemanticAnalyzer {
public:
    void analyze(const std::vector<std::shared_ptr<FuncNode>>& funcs);

private:
    std::unordered_map<std::string, TypeInfo> symbol_table_; //hashmap for symbols
    std::unordered_map<std::string, FuncInfo> function_table_; //hashmap for functions

    std::string current_func_;

    // Dimensions of the function that is being analyzed, resolved to numbers.
    std::unordered_map<std::string, int64_t> dim_bindings_;

    // Dimensions resolved at the call sites of each function. They outlive a
    // single analysis round, because a callee may be analyzed before the call
    // site that gives its dimensions is known.
    std::unordered_map<std::string, std::unordered_map<std::string, int64_t>>
        func_dim_bindings_;

    // Dimensions that were bound to two different numbers, reported once.
    std::unordered_set<std::string> reported_conflicts_;

    // Number of dimensions bound during the current analysis round.
    int new_bindings_ = 0;

    // Whether the current round reports errors and warnings. The rounds that
    // only collect dimension bindings stay silent.
    bool collect_errors_ = true;

    // Hard semantic errors gathered during the current analysis; if any are
    // present when analysis completes, analyze() throws a SemanticError.
    std::vector<std::string> errors_;

    // Record a recoverable hard error and continue analysis so all problems
    // can be reported together.
    void semantic_error(const std::string& msg);

    // Report a warning about the program, once the analysis has settled.
    void warn(const std::string& msg);

    void analyze_func(const std::shared_ptr<FuncNode>& func);
    void analyze_stmt(const std::shared_ptr<ASTNode>& stmt);
    void analyze_expr(const std::shared_ptr<ASTNode>& expr);
    void resolve_var(const std::string& name, int line, int col);
    void resolve_call(const std::string& name, size_t argc, int line, int col);
    bool is_tensor_expr(const std::shared_ptr<ASTNode>& expr);
    bool tensor_is_empty(const std::shared_ptr<ASTNode>& expr);
    bool is_uninitialized_tensor(const std::shared_ptr<ASTNode>& expr);
    void mark_initialized(const std::shared_ptr<ASTNode>& target);

    // Checks the data a tensor is initialized with against its declared dims.
    void check_tensor_init(const std::shared_ptr<TensorDeclNode>& node);

    // Resolves every dim of a shape to a number; false when one of them is
    // still symbolic.
    bool resolve_dims(const std::vector<ShapeDim>& shape, std::vector<int64_t>& out);

    // The dimensions that are known in the scope of the current function.
    std::unordered_map<std::string, int64_t> current_dim_bindings();

    // Binds the symbolic dimensions of a callee to the dimensions of the
    // arguments of a call; returns true when a new binding was found.
    void bind_call_dims(const std::shared_ptr<FunctionCallNode>& call);
    bool bind_dim(const std::string& func_name, const std::string& dim, int64_t value);

    // The concrete dims of an argument, when it is a tensor with a known shape.
    bool arg_shape(const std::shared_ptr<ASTNode>& arg, std::vector<int64_t>& out);

    // Puts the dimensions discovered at the call sites into the scope of a
    // function.
    void seed_dim_bindings(const std::string& func_name);
};

#endif
