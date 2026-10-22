#include "semantic.h"

namespace {

// The analysis is repeated until the dimensions of every function are known.
// A call site that appears late in the file can still bind the dimensions of a
// function that was analyzed first, so a few rounds are needed.
constexpr int kMaxAnalysisRounds = 8;

}  // namespace

void SemanticAnalyzer::semantic_error(const std::string& msg) {
    if (!collect_errors_) return;
    errors_.push_back(msg);
}

void SemanticAnalyzer::warn(const std::string& msg) {
    if (!collect_errors_) return;
    std::cerr << msg << std::endl;
}

void SemanticAnalyzer::analyze(const std::vector<std::shared_ptr<FuncNode>>& funcs) {
    errors_.clear();
    func_dim_bindings_.clear();
    reported_conflicts_.clear();

    // Build the function table first so calls can be resolved later.
    function_table_.clear();
    for (const auto& func : funcs) {
        if (!func) continue;
        FuncInfo info;
        info.name = func->name;
        info.param_count = func->params.size();
        info.returns = func->returns;
        info.is_main = (func->name == "main");
        for (const auto& param : func->params) {
            info.param_shapes.push_back(param->shape);
        }
        function_table_[func->name] = info;
    }

    // The dimensions of a function are discovered at its call sites, so the
    // analysis is repeated until no new dimension turns up. These rounds only
    // collect dimensions; the errors are reported by the last round.
    collect_errors_ = false;
    bool changed = true;
    for (int round = 0; round < kMaxAnalysisRounds && changed; ++round) {
        new_bindings_ = 0;
        for (const auto& func : funcs) {
            if (func) analyze_func(func);
        }
        changed = (new_bindings_ > 0);
    }

    // Report the errors of the last round, now that every dimension that can
    // be known is known.
    collect_errors_ = true;
    for (const auto& func : funcs) {
        if (func) analyze_func(func);
    }

    // Surface all collected hard errors at once. Throwing here stops the
    // pipeline before the AST/JSON/DOT are emitted for an invalid program.
    if (!errors_.empty()) {
        std::string combined;
        for (size_t i = 0; i < errors_.size(); ++i) {
            if (i != 0) combined += "\n";
            combined += errors_[i];
        }
        throw SemanticError(combined);
    }
}

void SemanticAnalyzer::analyze_func(const std::shared_ptr<FuncNode>& func) {
    current_func_ = func->name;

    // main must not declare parameters.
    if (func->name == "main" && !func->params.empty()) {
        warn("[Semantic Warning] 'main' should not take parameters; " +
             std::to_string(func->params.size()) + " found at L" +
             std::to_string(func->sp.line) + ":" + std::to_string(func->sp.col));
    }

    // Build the local symbol table from parameters.
    symbol_table_.clear();
    dim_bindings_.clear();
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

    // Put the dimensions that were found at the call sites into scope.
    seed_dim_bindings(func->name);

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
        warn("[Semantic Warning] 'main' does not have an explicit "
             "'return' statement; consider returning a result at L" +
             std::to_string(func->sp.line) + ":" + std::to_string(func->sp.col));
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
                // An integer assignment defines a compile-time constant, which
                // lets a symbolic dimension resolve to a number.
                Scalar s;
                auto& info = symbol_table_[name];
                // A tensor holds data, not a compile-time constant.
                if (!info.is_tensor && node->value &&
                    eval_scalar(node->value, s) && s.is_int) {
                    const int64_t value = static_cast<int64_t>(s.value);
                    if (info.const_value.has_value() && *info.const_value != value) {
                        semantic_error("[Semantic Error] '" + name +
                                       "' is assigned both " +
                                       std::to_string(*info.const_value) + " and " +
                                       std::to_string(value) + " at L" +
                                       std::to_string(node->sp.line) + ":" +
                                       std::to_string(node->sp.col));
                    } else {
                        info.const_value = value;
                    }
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
            if (node->init_value) {
                analyze_expr(node->init_value);
                // The data must fill the declared dims. A declaration without
                // a shape is a scalar, so there are no dims to fill.
                if (ti.is_tensor) {
                    check_tensor_init(node);
                }
            }
            break;
        }
        case ASTKind::RETURN: {
            auto node = std::static_pointer_cast<ReturnNode>(stmt);
            // Return is only permitted inside main.
            if (current_func_ != "main") {
                warn("[Semantic Warning] 'return' is only allowed inside "
                     "'main'; found in '" + current_func_ + "' at L" +
                     std::to_string(node->sp.line) + ":" +
                     std::to_string(node->sp.col));
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
            // The arguments give the dimensions of the callee.
            bind_call_dims(node);
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
                    const std::string loc =
                        " at L" + std::to_string(node->sp.line) + ":" +
                        std::to_string(node->sp.col);
                    const std::string arg = std::to_string(i + 1);
                    if (!is_tensor_expr(node->args[i])) {
                        semantic_error("[Semantic Error] '" + node->op +
                                       "' expects tensor argument(s), but argument " +
                                       arg + " is not a tensor" + loc);
                    } else if (tensor_is_empty(node->args[i])) {
                        semantic_error("[Semantic Error] '" + node->op +
                                       "' does not accept empty tensors, but argument " +
                                       arg + " has a zero-sized dimension" + loc);
                    } else if (is_uninitialized_tensor(node->args[i])) {
                        semantic_error("[Semantic Error] '" + node->op +
                                       "' does not accept empty tensors, but argument " +
                                       arg + " holds no data (never written)" + loc);
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
        warn("[Semantic Warning] Undeclared variable/parameter '" + name +
             "' at L" + std::to_string(line) + ":" + std::to_string(col));
    }
}

void SemanticAnalyzer::resolve_call(const std::string& name, size_t argc, int line, int col) {
    auto it = function_table_.find(name);
    if (it == function_table_.end()) {
        warn("[Semantic Warning] Call to undeclared function '" + name +
             "' at L" + std::to_string(line) + ":" + std::to_string(col));
        return;
    }
    if (it->second.param_count != argc) {
        warn("[Semantic Warning] Function '" + name + "' expects " +
             std::to_string(it->second.param_count) + " argument(s), got " +
             std::to_string(argc) + " at L" + std::to_string(line) + ":" +
             std::to_string(col));
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

void SemanticAnalyzer::check_tensor_init(const std::shared_ptr<TensorDeclNode>& node) {
    const auto bindings = current_dim_bindings();
    const TensorInitInfo info = analyze_tensor_init(node, &bindings);

    switch (info.status) {
        case TensorInitInfo::Status::SCALAR_INIT:
        case TensorInitInfo::Status::RAGGED:
        case TensorInitInfo::Status::RANK_MISMATCH:
        case TensorInitInfo::Status::COUNT_MISMATCH:
            semantic_error("[Semantic Error] " + info.message + " at L" +
                           std::to_string(node->sp.line) + ":" +
                           std::to_string(node->sp.col));
            break;
        default:
            // The initialiser is either absent, not a literal or not
            // statically known, so there is nothing to check against the dims.
            break;
    }
}

std::unordered_map<std::string, int64_t> SemanticAnalyzer::current_dim_bindings() {
    std::unordered_map<std::string, int64_t> bindings = dim_bindings_;
    for (const auto& entry : symbol_table_) {
        if (entry.second.const_value.has_value()) {
            bindings[entry.first] = *entry.second.const_value;
        }
    }
    return bindings;
}

bool SemanticAnalyzer::resolve_dims(const std::vector<ShapeDim>& shape,
                                    std::vector<int64_t>& out) {
    out.clear();
    const auto bindings = current_dim_bindings();
    for (const auto& dim : shape) {
        if (std::holds_alternative<int64_t>(dim)) {
            out.push_back(std::get<int64_t>(dim));
            continue;
        }
        const std::string& name = std::get<std::string>(dim);
        auto it = bindings.find(name);
        if (it == bindings.end()) return false;
        out.push_back(it->second);
    }
    return true;
}

void SemanticAnalyzer::bind_call_dims(const std::shared_ptr<FunctionCallNode>& call) {
    auto it = function_table_.find(call->name);
    if (it == function_table_.end()) return;

    const auto& param_shapes = it->second.param_shapes;
    for (size_t i = 0; i < call->args.size() && i < param_shapes.size(); ++i) {
        // Only an argument with a fully known shape can give a dimension.
        std::vector<int64_t> arg_dims;
        if (!arg_shape(call->args[i], arg_dims)) continue;

        const auto& param_shape = param_shapes[i];
        for (size_t d = 0; d < param_shape.size() && d < arg_dims.size(); ++d) {
            if (std::holds_alternative<std::string>(param_shape[d])) {
                bind_dim(call->name, std::get<std::string>(param_shape[d]),
                         arg_dims[d]);
            }
        }
    }
}

bool SemanticAnalyzer::bind_dim(const std::string& func_name, const std::string& dim,
                                 int64_t value) {
    auto& bindings = func_dim_bindings_[func_name];
    auto it = bindings.find(dim);
    if (it == bindings.end()) {
        bindings[dim] = value;
        new_bindings_++;
        return true;
    }
    if (it->second != value) {
        const std::string key = func_name + "." + dim;
        if (collect_errors_ && reported_conflicts_.insert(key).second) {
            semantic_error("[Semantic Error] dimension '" + dim + "' of '" +
                           func_name + "' is bound to both " +
                           std::to_string(it->second) + " and " +
                           std::to_string(value));
        }
    }
    return false;
}

bool SemanticAnalyzer::arg_shape(const std::shared_ptr<ASTNode>& arg,
                                 std::vector<int64_t>& out) {
    if (!arg || arg->kind() != ASTKind::VARIABLE) return false;
    auto node = std::static_pointer_cast<VariableNode>(arg);
    auto it = symbol_table_.find(node->name);
    if (it == symbol_table_.end()) return false;
    return resolve_dims(it->second.shape, out);
}

void SemanticAnalyzer::seed_dim_bindings(const std::string& func_name) {
    auto it = func_dim_bindings_.find(func_name);
    if (it == func_dim_bindings_.end()) return;

    for (const auto& entry : it->second) {
        const std::string& dim = entry.first;
        const int64_t value = entry.second;
        dim_bindings_[dim] = value;

        auto sym = symbol_table_.find(dim);
        if (sym == symbol_table_.end()) {
            TypeInfo ti;
            ti.const_value = value;
            symbol_table_[dim] = ti;
        } else if (!sym->second.const_value.has_value()) {
            sym->second.const_value = value;
        }
    }
}
