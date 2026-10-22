#include "json_visitor.h"

json JsonVisitor::shape_to_json(const std::vector<ShapeDim>& shape) {
    json j = json::array();
    for (const auto& dim : shape) {
        std::visit([&j](auto&& val) { j.push_back(val); }, dim);
    }
    return j;
}

//Attaches these line and column of node's beginning in source code.
void JsonVisitor::add_span(json& j, const span& sp) {
    j["line"] = sp.line;
    j["col"] = sp.col;
}

std::any JsonVisitor::visitFuncNode(std::shared_ptr<FuncNode> node) {
    json j;
    j["funcname"] = node->name;
    add_span(j, node->sp);

    json params_array = json::array();
    for (auto& param : node->params) {
        json param_json;
        param_json["kind"] = "Parameter";
        param_json["name"] = param->name;
        param_json["dtype"] = param->dtype;
        param_json["shape"] = shape_to_json(param->shape);
        add_span(param_json, param->sp);
        params_array.push_back(param_json);
    }
    j["params"] = params_array;

    json returns_array = json::array();
    for (const auto& ret : node->returns) {
        returns_array.push_back(ret);
    }
    j["returns"] = returns_array;

    json body_array = json::array();
    for (const auto& stmt : node->body) {
        body_array.push_back(std::any_cast<json>(stmt->accept(*this)));
    }
    j["body"] = body_array;

    return j;
}

std::any JsonVisitor::visitParamNode(std::shared_ptr<ParamNode> node) {
    json j;
    j["kind"] = "Parameter";
    j["name"] = node->name;
    j["dtype"] = node->dtype;
    j["shape"] = shape_to_json(node->shape);
    add_span(j, node->sp);
    return j;
}

std::any JsonVisitor::visitForLoopNode(std::shared_ptr<ForLoopNode> node) {
    json j;
    j["kind"] = "For Loop";
    j["var"] = node->var;
    j["bound"] = node->bound;
    j["type"] = node->attr_type;
    j["size"] = node->tile_or_unroll;
    add_span(j, node->sp);

    json body_array = json::array();
    for (const auto& stmt : node->body) {
        body_array.push_back(std::any_cast<json>(stmt->accept(*this)));
    }
    j["body"] = body_array;
    return j;
}

std::any JsonVisitor::visitIfNode(std::shared_ptr<IfNode> node) {
    json j;
    j["kind"] = "If";
    add_span(j, node->sp);
    j["condition"] = std::any_cast<json>(node->condition->accept(*this));

    json body_array = json::array();
    for (const auto& stmt : node->body) {
        body_array.push_back(std::any_cast<json>(stmt->accept(*this)));
    }
    j["body"] = body_array;

    json orelse_array = json::array();
    for (const auto& stmt : node->orelse) {
        orelse_array.push_back(std::any_cast<json>(stmt->accept(*this)));
    }
    j["orelse"] = orelse_array;
    return j;
}

std::any JsonVisitor::visitAssignNode(std::shared_ptr<AssignNode> node) {
    json j;
    j["kind"] = "Assign";
    add_span(j, node->sp);
    j["target"] = std::any_cast<json>(node->target->accept(*this));
    j["op"] = node->op;
    j["value"] = std::any_cast<json>(node->value->accept(*this));
    return j;
}

std::any JsonVisitor::visitTensorDeclNode(std::shared_ptr<TensorDeclNode> node) {
    json j;
    j["kind"] = "Tensor";
    j["name"] = node->name;
    j["dtype"] = node->dtype;
    j["shape"] = shape_to_json(node->shape);
    add_span(j, node->sp);

    bool has_symbolic = false;
    for (const auto& d : node->shape) {
        if (std::holds_alternative<std::string>(d)) { has_symbolic = true; break; }
    }
    if (has_symbolic) {
        j["data"] = nullptr;
    } else if (node->init_value) {
        j["data"] = std::any_cast<json>(node->init_value->accept(*this));
    } else {
        j["data"] = nullptr;
    }
    // The values of the initialiser, nested the way the literal nests. They
    // are only known when every element of the initialiser is a compile-time
    // constant.
    j["data_values"] = has_symbolic ? json(nullptr) : tensor_values(node);
    return j;
}

// The values held by the initialiser of a tensor declaration.
json JsonVisitor::tensor_values(const std::shared_ptr<TensorDeclNode>& node) {
    const TensorInitInfo info = analyze_tensor_init(node, nullptr);
    if (info.status != TensorInitInfo::Status::OK) return json(nullptr);
    return rebuild_tensor(info.literal_shape, info.raw);
}

std::any JsonVisitor::visitBinaryOpNode(std::shared_ptr<BinaryOpNode> node) {
    json j;
    j["kind"] = "Binary";
    j["op"] = node->op;
    add_span(j, node->sp);
    j["left"] = std::any_cast<json>(node->left->accept(*this));
    j["right"] = std::any_cast<json>(node->right->accept(*this));
    return j;
}

std::any JsonVisitor::visitUnaryOpNode(std::shared_ptr<UnaryOpNode> node) {
    json j;
    j["kind"] = "UnaryOp";
    j["op"] = node->op;
    add_span(j, node->sp);
    if (node->operand) {
        j["operand"] = std::any_cast<json>(node->operand->accept(*this));
    }
    return j;
}

std::any JsonVisitor::visitSubscriptNode(std::shared_ptr<SubscriptNode> node) {
    json j;
    j["kind"] = "Subscript";
    j["value"] = node->value;
    add_span(j, node->sp);

    json indices_array = json::array();
    for (const auto& idx : node->indices) {
        if (idx) {
            indices_array.push_back(std::any_cast<json>(idx->accept(*this)));
        }
    }
    j["indices"] = indices_array;
    return j;
}

std::any JsonVisitor::visitBoolOpNode(std::shared_ptr<BoolOpNode> node) {
    json j;
    j["kind"] = "BoolOp";
    j["op"] = node->op;
    add_span(j, node->sp);

    json values_array = json::array();
    for (const auto& val : node->values) {
        values_array.push_back(std::any_cast<json>(val->accept(*this)));
    }
    j["values"] = values_array;
    return j;
}

std::any JsonVisitor::visitTernaryNode(std::shared_ptr<TernaryNode> node) {
    json j;
    j["kind"] = "Ternary";
    add_span(j, node->sp);
    j["condition"] = std::any_cast<json>(node->cond->accept(*this));
    j["body"] = std::any_cast<json>(node->body->accept(*this));
    j["orelse"] = std::any_cast<json>(node->orelse->accept(*this));
    return j;
}

std::any JsonVisitor::visitVariableNode(std::shared_ptr<VariableNode> node) {
    json j;
    j["kind"] = "Variable";
    j["name"] = node->name;
    add_span(j, node->sp);
    return j;
}

std::any JsonVisitor::visitConstantNode(std::shared_ptr<ConstantNode> node) {
    json j;
    j["kind"] = "Constant";
    add_span(j, node->sp);
    if (node->is_integer) {
        j["value"] = static_cast<int64_t>(node->value);
    } else {
        j["value"] = node->value;
    }
    return j;
}

std::any JsonVisitor::visitReturnNode(std::shared_ptr<ReturnNode> node) {
    json j;
    j["kind"] = "Return";
    add_span(j, node->sp);
    if (node->value) {
        j["value"] = std::any_cast<json>(node->value->accept(*this));
    } else {
        j["value"] = nullptr;
    }
    return j;
}

std::any JsonVisitor::visitBreakNode(std::shared_ptr<BreakNode> node) {
    json j;
    j["kind"] = "Break";
    add_span(j, node->sp);
    return j;
}

std::any JsonVisitor::visitContinueNode(std::shared_ptr<ContinueNode> node) {
    json j;
    j["kind"] = "Continue";
    add_span(j, node->sp);
    return j;
}

std::any JsonVisitor::visitListNode(std::shared_ptr<ListNode> node) {
    json j;
    j["kind"] = "List";
    add_span(j, node->sp);

    json elements_array = json::array();
    for (const auto& elem : node->elements) {
        elements_array.push_back(std::any_cast<json>(elem->accept(*this)));
    }
    j["elements"] = elements_array;
    return j;
}

std::any JsonVisitor::visitBuiltinCallNode(std::shared_ptr<BuiltinCallNode> node) {
    json j;
    j["kind"] = "BuiltinCall";
    j["op"] = node->op;
    add_span(j, node->sp);

    json args_array = json::array();
    for (const auto& arg : node->args) {
        args_array.push_back(std::any_cast<json>(arg->accept(*this)));
    }
    j["args"] = args_array;

    return j;
}

std::any JsonVisitor::visitPrintNode(std::shared_ptr<PrintNode> node) {
    json j;
    j["kind"] = "Print";
    add_span(j, node->sp);
    if (node->value) {
        j["value"] = std::any_cast<json>(node->value->accept(*this));
    } else {
        j["value"] = nullptr;
    }
    return j;
}

std::any JsonVisitor::visitFunctionCallNode(std::shared_ptr<FunctionCallNode> node) {
    json j;
    j["kind"] = "FunctionCall";
    j["name"] = node->name;
    add_span(j, node->sp);

    json args_array = json::array();
    for (const auto& arg : node->args) {
        args_array.push_back(std::any_cast<json>(arg->accept(*this)));
    }
    j["args"] = args_array;

    return j;
}
