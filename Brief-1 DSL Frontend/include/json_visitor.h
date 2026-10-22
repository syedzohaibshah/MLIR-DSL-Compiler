#ifndef TLDR_JSON_VISITOR_H
#define TLDR_JSON_VISITOR_H

#include "ast_nodes.h"
#include "const_eval.h"
#include "json.hpp"
#include <string>

using json = nlohmann::json;

class JsonVisitor : public Visitor {
public:
    std::any visitFuncNode(std::shared_ptr<FuncNode> node) override;
    std::any visitParamNode(std::shared_ptr<ParamNode> node) override;
    std::any visitForLoopNode(std::shared_ptr<ForLoopNode> node) override;
    std::any visitIfNode(std::shared_ptr<IfNode> node) override;
    std::any visitAssignNode(std::shared_ptr<AssignNode> node) override;
    std::any visitTensorDeclNode(std::shared_ptr<TensorDeclNode> node) override;
    std::any visitBinaryOpNode(std::shared_ptr<BinaryOpNode> node) override;
    std::any visitUnaryOpNode(std::shared_ptr<UnaryOpNode> node) override;
    std::any visitSubscriptNode(std::shared_ptr<SubscriptNode> node) override;
    std::any visitBoolOpNode(std::shared_ptr<BoolOpNode> node) override;
    std::any visitTernaryNode(std::shared_ptr<TernaryNode> node) override;
    std::any visitVariableNode(std::shared_ptr<VariableNode> node) override;
    std::any visitConstantNode(std::shared_ptr<ConstantNode> node) override;
    std::any visitReturnNode(std::shared_ptr<ReturnNode> node) override;
    std::any visitBreakNode(std::shared_ptr<BreakNode> node) override;
    std::any visitContinueNode(std::shared_ptr<ContinueNode> node) override;
    std::any visitListNode(std::shared_ptr<ListNode> node) override;
    std::any visitBuiltinCallNode(std::shared_ptr<BuiltinCallNode> node) override;
    std::any visitPrintNode(std::shared_ptr<PrintNode> node) override;
    std::any visitFunctionCallNode(std::shared_ptr<FunctionCallNode> node) override;

private:
    json shape_to_json(const std::vector<ShapeDim>& shape);
    void add_span(json& j, const span& sp);
    json tensor_values(const std::shared_ptr<TensorDeclNode>& node);
};

#endif
