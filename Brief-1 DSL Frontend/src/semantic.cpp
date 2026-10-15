#include "semantic.h"

void SemanticAnalyzer::analyze(const std::vector<std::shared_ptr<FuncNode>>& funcs) {
    // Build the function table first so calls can be resolved later.
    function_table_.clear();
    for (const auto& func : funcs) {
        if (!func) continue;
        FuncInfo info;
        info.name = func->name;
        info.param_count = func->params.size();
        info.returns = func->returns;
        info.is_main = (func->name == "main");
        function_table_[func->name] = info;
    }

    for (const auto& func : funcs) {
        if (func) analyze_func(func);
    }
}

void SemanticAnalyzer::analyze_func(const std::shared_ptr<FuncNode>& func) {
    current_func_ = func->name;

    // main must not declare parameters.
    if (func->name == "main" && !func->params.empty()) {
        std::cerr << "[Semantic Warning] 'main' should not take parameters; "
                  << func->params.size() << " found at L" << func->sp.line << ":"
                  << func->sp.col << std::endl;
    }

    // Build the local symbol table from parameters.
    symbol_table_.clear();
    for (const auto& param : func->params) {
        TypeInfo ti;
        ti.is_tensor = !param->shape.empty();
        ti.is_param = true;
        // Parameters carry data provided by the caller.
        ti.initialized = true;
        ti.dtype = param->dtype;
        ti.shape = param->shape;
        symbol_table_[param->name] = ti;
        // Symbolic shape dimensions act as compile-time constants in scope.
        for (const auto& dim : param->shape) {
            if (std::holds_alternative<std::string>(dim)) {
                symbol_table_[std::get<std::string>(dim)] = TypeInfo{};
            }
        }
    }

    // Names declared by the return annotation are output symbols of this scope.
    // Keep the parameter's TypeInfo if the returned symbol is also a parameter.
    for (const auto& out : func->returns) {
        if (symbol_table_.find(out) == symbol_table_.end()) {
            symbol_table_[out] = TypeInfo{};
        }
    }

    // Track whether this function produces a tensor result so tensor checks
    // can accept calls to tensor-returning functions.
    function_table_[func->name].returns_tensor = false;
    for (const auto& out : func->returns) {
        auto it = symbol_table_.find(out);
        if (it != symbol_table_.end() && it->second.is_tensor) {
            function_table_[func->name].returns_tensor = true;
            break;
        }
    }

    // Reset the explicit-return flag for this function.
    function_table_[func->name].has_explicit_return = false;

    for (const auto& stmt : func->body) {
        if (stmt) analyze_stmt(stmt);
    }

    if (func->name == "main" && !function_table_[func->name].has_explicit_return) {
        std::cerr << "[Semantic Warning] 'main' does not have an explicit "
                  << "'return' statement; consider returning a result at L"
                  << func->sp.line << ":" << func->sp.col << std::endl;
    }
}

void SemanticAnalyzer::analyze_stmt(const std::shared_ptr<ASTNode>& stmt) {
    if (!stmt) return;

    switch (stmt->kind()) {
        case ASTKind::FOR_LOOP: {
            auto node = std::static_pointer_cast<ForLoopNode>(stmt);
            symbol_table_[node->var] = TypeInfo{}; // loop variable
            for (const auto& s : node->body) analyze_stmt(s);
            break;
        }
        case ASTKind::IF: {
            auto node = std::static_pointer_cast<IfNode>(stmt);
            analyze_expr(node->condition);
            for (const auto& s : node->body) analyze_stmt(s);
            for (const auto& s : node->orelse) analyze_stmt(s);
            break;
        }
        case ASTKind::ASSIGN: {
            auto node = std::static_pointer_cast<AssignNode>(stmt);
            // A plain variable target introduces a new variable unless it
            // already exists; keep the existing declaration's TypeInfo (e.g. a
            // tensor buffer reassigned with a kernel result).
            if (node->target && node->target->kind() == ASTKind::VARIABLE) {
                const auto& name =
                    std::static_pointer_cast<VariableNode>(node->target)->name;
                if (symbol_table_.find(name) == symbol_table_.end()) {
                    symbol_table_[name] = TypeInfo{}; // implicit declaration
                }
            } else if (node->target) {
                analyze_expr(node->target);
            }
            analyze_expr(node->value);
            // The target now holds data (whole-tensor or element writes).
            mark_initialized(node->target);
            break;
        }
        case ASTKind::TENSOR_DECL: {
            auto node = std::static_pointer_cast<TensorDeclNode>(stmt);
            TypeInfo ti;
            // A declaration is only a tensor when it carries a shape;
            // bare dtype decls (e.g. "x : f32") are scalars.
            ti.is_tensor = !node->shape.empty();
            ti.dtype = node->dtype;
            ti.shape = node->shape;
            // Holds data when a value was provided at declaration.
            ti.initialized = (node->init_value != nullptr);
            symbol_table_[node->name] = ti;
            if (node->init_value) analyze_expr(node->init_value);
            break;
        }
        case ASTKind::RETURN: {
            auto node = std::static_pointer_cast<ReturnNode>(stmt);
            // Return is only permitted inside main.
            if (current_func_ != "main") {
                std::cerr << "[Semantic Warning] 'return' is only allowed inside "
                          << "'main'; found in '" << current_func_ << "' at L"
                          << node->sp.line << ":" << node->sp.col << std::endl;
            } else {
                function_table_[current_func_].has_explicit_return = true;
            }
            if (node->value) analyze_expr(node->value);
            break;
        }
        case ASTKind::PRINT: {
            auto node = std::static_pointer_cast<PrintNode>(stmt);
            if (node->value) analyze_expr(node->value);
            break;
        }
        case ASTKind::BREAK:
        case ASTKind::CONTINUE:
            break;
        default:
            break;
    }
}

void SemanticAnalyzer::analyze_expr(const std::shared_ptr<ASTNode>& expr) {
    if (!expr) return;

    switch (expr->kind()) {
        case ASTKind::SUBSCRIPT: {
            auto node = std::static_pointer_cast<SubscriptNode>(expr);
            resolve_var(node->value, node->sp.line, node->sp.col);
            for (const auto& idx : node->indices) analyze_expr(idx);
            break;
        }
        case ASTKind::BINARY_OP: {
            auto node = std::static_pointer_cast<BinaryOpNode>(expr);
            analyze_expr(node->left);
            analyze_expr(node->right);
            break;
        }
        case ASTKind::UNARY_OP: {
            auto node = std::static_pointer_cast<UnaryOpNode>(expr);
            analyze_expr(node->operand);
            break;
        }
        case ASTKind::BOOL_OP: {
            auto node = std::static_pointer_cast<BoolOpNode>(expr);
            for (const auto& v : node->values) analyze_expr(v);
            break;
        }
        case ASTKind::VARIABLE: {
            auto node = std::static_pointer_cast<VariableNode>(expr);
            resolve_var(node->name, node->sp.line, node->sp.col);
            break;
        }
        case ASTKind::FUNC_CALL: {
            auto node = std::static_pointer_cast<FunctionCallNode>(expr);
            for (const auto& arg : node->args) analyze_expr(arg);
            resolve_call(node->name, node->args.size(), node->sp.line, node->sp.col);
            break;
        }
        case ASTKind::CUSTOM_OP: {
            auto node = std::static_pointer_cast<BuiltinCallNode>(expr);
            for (const auto& arg : node->args) analyze_expr(arg);
            // Operators declared in the registry as tensor-only (softmax, relu,
            // ...) accept non-empty tensor arguments. Unknown ops fall back to
            // the same check for backwards compatibility.
            const BuiltinOpSpec* spec = find_builtin_op(node->op);
            if (!spec || spec->requires_tensor_args) {
                for (size_t i = 0; i < node->args.size(); i++) {
                    if (!is_tensor_expr(node->args[i])) {
                        std::cerr << "[Semantic Error] '" << node->op
                                  << "' expects tensor argument(s), but argument "
                                  << (i + 1) << " is not a tensor at L"
                                  << node->sp.line << ":" << node->sp.col << std::endl;
                    } else if (tensor_is_empty(node->args[i])) {
                        std::cerr << "[Semantic Error] '" << node->op
                                  << "' does not accept empty tensors, but argument "
                                  << (i + 1) << " has a zero-sized dimension at L"
                                  << node->sp.line << ":" << node->sp.col << std::endl;
                    } else if (is_uninitialized_tensor(node->args[i])) {
                        std::cerr << "[Semantic Error] '" << node->op
                                  << "' does not accept empty tensors, but argument "
                                  << (i + 1) << " holds no data (never written) at L"
                                  << node->sp.line << ":" << node->sp.col << std::endl;
                    }
                }
            }
            break;
        }
        case ASTKind::LIST: {
            auto node = std::static_pointer_cast<ListNode>(expr);
            for (const auto& e : node->elements) analyze_expr(e);
            break;
        }
        case ASTKind::TENSOR_DECL:
        case ASTKind::CONSTANT:
        case ASTKind::TERNARY:
            break;
        default:
            break;
    }
}

void SemanticAnalyzer::resolve_var(const std::string& name, int line, int col) {
    if (symbol_table_.find(name) == symbol_table_.end()) {
        std::cerr << "[Semantic Warning] Undeclared variable/parameter '" << name
                  << "' at L" << line << ":" << col << std::endl;
    }
}

void SemanticAnalyzer::resolve_call(const std::string& name, size_t argc, int line, int col) {
    auto it = function_table_.find(name);
    if (it == function_table_.end()) {
        std::cerr << "[Semantic Warning] Call to undeclared function '" << name
                  << "' at L" << line << ":" << col << std::endl;
        return;
    }
    if (it->second.param_count != argc) {
        std::cerr << "[Semantic Warning] Function '" << name << "' expects "
                  << it->second.param_count << " argument(s), got " << argc
                  << " at L" << line << ":" << col << std::endl;
    }
}

bool SemanticAnalyzer::tensor_is_empty(const std::shared_ptr<ASTNode>& expr) {
    if (!expr) return false;

    // Only statically-sized tensors can be checked; a zero-sized shape dim
    // means the tensor holds no elements.
    if (expr->kind() == ASTKind::VARIABLE) {
        auto node = std::static_pointer_cast<VariableNode>(expr);
        auto it = symbol_table_.find(node->name);
        if (it == symbol_table_.end()) return false;
        for (const auto& dim : it->second.shape) {
            if (std::holds_alternative<int64_t>(dim) &&
                std::get<int64_t>(dim) == 0) {
                return true;
            }
        }
    }
    return false;
}

bool SemanticAnalyzer::is_uninitialized_tensor(const std::shared_ptr<ASTNode>& expr) {
    if (!expr) return false;

    if (expr->kind() == ASTKind::VARIABLE) {
        auto node = std::static_pointer_cast<VariableNode>(expr);
        auto it = symbol_table_.find(node->name);
        if (it != symbol_table_.end()) {
            return it->second.is_tensor && !it->second.initialized;
        }
    }
    return false;
}

void SemanticAnalyzer::mark_initialized(const std::shared_ptr<ASTNode>& target) {
    if (!target) return;

    std::string base;
    if (target->kind() == ASTKind::VARIABLE) {
        base = std::static_pointer_cast<VariableNode>(target)->name;
    } else if (target->kind() == ASTKind::SUBSCRIPT) {
        base = std::static_pointer_cast<SubscriptNode>(target)->value;
    }

    if (!base.empty()) {
        auto it = symbol_table_.find(base);
        if (it != symbol_table_.end()) it->second.initialized = true;
    }
}

bool SemanticAnalyzer::is_tensor_expr(const std::shared_ptr<ASTNode>& expr) {
    if (!expr) return false;

    switch (expr->kind()) {
        case ASTKind::VARIABLE: {
            auto node = std::static_pointer_cast<VariableNode>(expr);
            auto it = symbol_table_.find(node->name);
            if (it != symbol_table_.end()) return it->second.is_tensor;
            return false;
        }
        case ASTKind::SUBSCRIPT:
            // Subscripting a tensor yields a scalar element, not a tensor.
            return false;
        case ASTKind::CUSTOM_OP: {
            // Builtin tensor ops return tensors (see the registry).
            const BuiltinOpSpec* spec = find_builtin_op(
                std::static_pointer_cast<BuiltinCallNode>(expr)->op);
            return spec ? spec->returns_tensor : true;
        }
        case ASTKind::FUNC_CALL: {
            auto node = std::static_pointer_cast<FunctionCallNode>(expr);
            auto it = function_table_.find(node->name);
            if (it != function_table_.end()) return it->second.returns_tensor;
            return false;
        }
        case ASTKind::TENSOR_DECL:
            return true;
        default:
            return false;
    }
}
