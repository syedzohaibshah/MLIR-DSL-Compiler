#include "ast_nodes.h"

std::string indent_str(int depth) {
    return std::string(depth * 2, ' ');
}

std::any FuncNode::accept(Visitor& visitor) { return visitor.visitFuncNode(std::static_pointer_cast<FuncNode>(shared_from_this())); }
std::any ParamNode::accept(Visitor& visitor) { return visitor.visitParamNode(std::static_pointer_cast<ParamNode>(shared_from_this())); }
std::any ForLoopNode::accept(Visitor& visitor) { return visitor.visitForLoopNode(std::static_pointer_cast<ForLoopNode>(shared_from_this())); }
std::any IfNode::accept(Visitor& visitor) { return visitor.visitIfNode(std::static_pointer_cast<IfNode>(shared_from_this())); }
std::any AssignNode::accept(Visitor& visitor) { return visitor.visitAssignNode(std::static_pointer_cast<AssignNode>(shared_from_this())); }
std::any TensorDeclNode::accept(Visitor& visitor) { return visitor.visitTensorDeclNode(std::static_pointer_cast<TensorDeclNode>(shared_from_this())); }
std::any BinaryOpNode::accept(Visitor& visitor) { return visitor.visitBinaryOpNode(std::static_pointer_cast<BinaryOpNode>(shared_from_this())); }
std::any UnaryOpNode::accept(Visitor& visitor) { return visitor.visitUnaryOpNode(std::static_pointer_cast<UnaryOpNode>(shared_from_this())); }
std::any SubscriptNode::accept(Visitor& visitor) { return visitor.visitSubscriptNode(std::static_pointer_cast<SubscriptNode>(shared_from_this())); }
std::any BoolOpNode::accept(Visitor& visitor) { return visitor.visitBoolOpNode(std::static_pointer_cast<BoolOpNode>(shared_from_this())); }
std::any TernaryNode::accept(Visitor& visitor) { return visitor.visitTernaryNode(std::static_pointer_cast<TernaryNode>(shared_from_this())); }
std::any VariableNode::accept(Visitor& visitor) { return visitor.visitVariableNode(std::static_pointer_cast<VariableNode>(shared_from_this())); }
std::any ConstantNode::accept(Visitor& visitor) { return visitor.visitConstantNode(std::static_pointer_cast<ConstantNode>(shared_from_this())); }
std::any ReturnNode::accept(Visitor& visitor) { return visitor.visitReturnNode(std::static_pointer_cast<ReturnNode>(shared_from_this())); }
std::any BreakNode::accept(Visitor& visitor) { return visitor.visitBreakNode(std::static_pointer_cast<BreakNode>(shared_from_this())); }
std::any ContinueNode::accept(Visitor& visitor) { return visitor.visitContinueNode(std::static_pointer_cast<ContinueNode>(shared_from_this())); }
std::any ListNode::accept(Visitor& visitor) { return visitor.visitListNode(std::static_pointer_cast<ListNode>(shared_from_this())); }
std::any BuiltinCallNode::accept(Visitor& visitor) { return visitor.visitBuiltinCallNode(std::static_pointer_cast<BuiltinCallNode>(shared_from_this())); }
std::any PrintNode::accept(Visitor& visitor) { return visitor.visitPrintNode(std::static_pointer_cast<PrintNode>(shared_from_this())); }
std::any FunctionCallNode::accept(Visitor& visitor) { return visitor.visitFunctionCallNode(std::static_pointer_cast<FunctionCallNode>(shared_from_this())); }

std::any Visitor::visitFuncNode(std::shared_ptr<FuncNode>) { return {}; }
std::any Visitor::visitParamNode(std::shared_ptr<ParamNode>) { return {}; }
std::any Visitor::visitForLoopNode(std::shared_ptr<ForLoopNode>) { return {}; }
std::any Visitor::visitIfNode(std::shared_ptr<IfNode>) { return {}; }
std::any Visitor::visitAssignNode(std::shared_ptr<AssignNode>) { return {}; }
std::any Visitor::visitTensorDeclNode(std::shared_ptr<TensorDeclNode>) { return {}; }
std::any Visitor::visitBinaryOpNode(std::shared_ptr<BinaryOpNode>) { return {}; }
std::any Visitor::visitUnaryOpNode(std::shared_ptr<UnaryOpNode>) { return {}; }
std::any Visitor::visitSubscriptNode(std::shared_ptr<SubscriptNode>) { return {}; }
std::any Visitor::visitBoolOpNode(std::shared_ptr<BoolOpNode>) { return {}; }
std::any Visitor::visitTernaryNode(std::shared_ptr<TernaryNode>) { return {}; }
std::any Visitor::visitVariableNode(std::shared_ptr<VariableNode>) { return {}; }
std::any Visitor::visitConstantNode(std::shared_ptr<ConstantNode>) { return {}; }
std::any Visitor::visitReturnNode(std::shared_ptr<ReturnNode>) { return {}; }
std::any Visitor::visitBreakNode(std::shared_ptr<BreakNode>) { return {}; }
std::any Visitor::visitContinueNode(std::shared_ptr<ContinueNode>) { return {}; }
std::any Visitor::visitListNode(std::shared_ptr<ListNode>) { return {}; }
std::any Visitor::visitBuiltinCallNode(std::shared_ptr<BuiltinCallNode>) { return {}; }
std::any Visitor::visitPrintNode(std::shared_ptr<PrintNode>) { return {}; }
std::any Visitor::visitFunctionCallNode(std::shared_ptr<FunctionCallNode>) { return {}; }

static void print_shape(const std::vector<ShapeDim>& shape, std::ostream& os) {
    if (shape.empty()) return;
    os << "(";
    for (size_t i = 0; i < shape.size(); i++) {
        if (i > 0) os << ", ";
        if (std::holds_alternative<int64_t>(shape[i]))
            os << std::get<int64_t>(shape[i]);
        else
            os << std::get<std::string>(shape[i]);
    }
    os << ")";
}

void FuncNode::print(int indent) const {
    std::cout << indent_str(indent) << "FuncNode(name=" << name;
    if (!returns.empty()) {
        std::cout << " -> [";
        for (size_t i = 0; i < returns.size(); i++) {
            if (i > 0) std::cout << ", ";
            std::cout << returns[i];
        }
        std::cout << "]";
    }
    std::cout << ")\n";
    std::cout << indent_str(indent + 1) << "params:\n";
    for (auto& p : params) p->print(indent + 2);
    std::cout << indent_str(indent + 1) << "body:\n";
    for (auto& s : body) s->print(indent + 2);
}

void ParamNode::print(int indent) const {
    std::cout << indent_str(indent) << "ParamNode(name=" << name << ", dtype=" << dtype;
    print_shape(shape, std::cout);
    std::cout << ")\n";
}

void ForLoopNode::print(int indent) const {
    std::cout << indent_str(indent) << "ForLoopNode(var=" << var << ", bound=" << bound;
    if (!attr_type.empty())
        std::cout << ", " << attr_type << "=" << tile_or_unroll;
    std::cout << ")\n";
    for (auto& s : body) s->print(indent + 1);
}

void IfNode::print(int indent) const {
    std::cout << indent_str(indent) << "IfNode(\n";
    std::cout << indent_str(indent + 1) << "condition:\n";
    condition->print(indent + 2);
    std::cout << indent_str(indent + 1) << "body:\n";
    for (auto& s : body) s->print(indent + 2);
    if (!orelse.empty()) {
        std::cout << indent_str(indent + 1) << "else:\n";
        for (auto& s : orelse) s->print(indent + 2);
    }
    std::cout << indent_str(indent) << ")\n";
}

void AssignNode::print(int indent) const {
    std::cout << indent_str(indent) << "AssignNode(op=" << op << ")\n";
    std::cout << indent_str(indent + 1) << "target:\n";
    target->print(indent + 2);
    std::cout << indent_str(indent + 1) << "value:\n";
    value->print(indent + 2);
}

void TensorDeclNode::print(int indent) const {
    std::cout << indent_str(indent) << "TensorDeclNode(name=" << name << ", dtype=" << dtype;
    print_shape(shape, std::cout);
    std::cout << ")\n";
    if (init_value) {
        std::cout << indent_str(indent + 1) << "init:\n";
        init_value->print(indent + 2);
    }
}

void BinaryOpNode::print(int indent) const {
    std::cout << indent_str(indent) << "BinaryOpNode(op=" << op << ")\n";
    left->print(indent + 1);
    right->print(indent + 1);
}

void UnaryOpNode::print(int indent) const {
    std::cout << indent_str(indent) << "UnaryOpNode(op=" << op << ")\n";
    operand->print(indent + 1);
}

void SubscriptNode::print(int indent) const {
    std::cout << indent_str(indent) << "SubscriptNode(value=" << value << ")\n";
    for (auto& idx : indices) idx->print(indent + 1);
}

void BoolOpNode::print(int indent) const {
    std::cout << indent_str(indent) << "BoolOpNode(op=" << op << ")\n";
    for (auto& v : values) v->print(indent + 1);
}

void TernaryNode::print(int indent) const {
    std::cout << indent_str(indent) << "TernaryNode(\n";
    std::cout << indent_str(indent + 1) << "condition:\n";
    cond->print(indent + 2);
    std::cout << indent_str(indent + 1) << "body:\n";
    body->print(indent + 2);
    std::cout << indent_str(indent + 1) << "else:\n";
    orelse->print(indent + 2);
    std::cout << indent_str(indent) << ")\n";
}

void VariableNode::print(int indent) const {
    std::cout << indent_str(indent) << "VariableNode(name=" << name << ")\n";
}

void ConstantNode::print(int indent) const {
    std::cout << indent_str(indent) << "ConstantNode(value=" << value << ")\n";
}

void ReturnNode::print(int indent) const {
    if (value) {
        std::cout << indent_str(indent) << "ReturnNode(\n";
        value->print(indent + 1);
        std::cout << indent_str(indent) << ")\n";
    } else {
        std::cout << indent_str(indent) << "ReturnNode(None)\n";
    }
}

void BreakNode::print(int indent) const {
    std::cout << indent_str(indent) << "BreakNode()\n";
}

void ContinueNode::print(int indent) const {
    std::cout << indent_str(indent) << "ContinueNode()\n";
}

void ListNode::print(int indent) const {
    std::cout << indent_str(indent) << "ListNode(\n";
    for (auto& e : elements) e->print(indent + 1);
    std::cout << indent_str(indent) << ")\n";
}

void BuiltinCallNode::print(int indent) const{
    std::cout << indent_str(indent) << "Custom_OP()\n";
};

void PrintNode::print(int indent) const {
    std::cout << indent_str(indent) << "PrintNode(\n";
    if (value) {
        value->print(indent + 1);
    }
    std::cout << indent_str(indent) << ")\n";
}

void FunctionCallNode::print(int indent) const {
    std::cout << indent_str(indent) << "FunctionCallNode(name=" << name << ")\n";
    for (auto& a : args) a->print(indent + 1);
}