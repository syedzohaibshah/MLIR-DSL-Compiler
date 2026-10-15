#ifndef TLDR_AST_NODES_H
#define TLDR_AST_NODES_H

#include <string>
#include <vector>
#include <memory>
#include <variant>
#include <any>
#include <iostream>
#include <sstream>

using ShapeDim = std::variant<int64_t, std::string>;
struct span
{
    int line;
    int col;
};

enum class ASTKind {
    FUNC, PARAM, FOR_LOOP, IF, ASSIGN, TENSOR_DECL,
    BINARY_OP, UNARY_OP, SUBSCRIPT, BOOL_OP, TERNARY,
    VARIABLE, CONSTANT, RETURN, BREAK, CONTINUE, LIST, CUSTOM_OP,
    PRINT, FUNC_CALL
};

class Visitor;

struct ASTNode : public std::enable_shared_from_this<ASTNode> {
    span sp;
    virtual ~ASTNode() = default;
    virtual ASTKind kind() const = 0;
    virtual void print(int indent = 0) const = 0;
    virtual std::any accept(Visitor& visitor) = 0;
};

struct FuncNode : ASTNode {
    std::string name;
    std::vector<std::shared_ptr<struct ParamNode>> params;
    std::vector<std::string> returns;
    std::vector<std::shared_ptr<ASTNode>> body;
    ASTKind kind() const override { return ASTKind::FUNC; }
    void print(int indent) const override;
    std::any accept(Visitor& visitor) override;
};

struct ParamNode : ASTNode {
    std::string name;
    std::string dtype;
    std::vector<ShapeDim> shape;
    ASTKind kind() const override { return ASTKind::PARAM; }
    void print(int indent) const override;
    std::any accept(Visitor& visitor) override;
};

struct ForLoopNode : ASTNode {
    std::string var;
    std::string bound;
    int tile_or_unroll = 0;
    std::string attr_type;
    std::vector<std::shared_ptr<ASTNode>> body;
    ASTKind kind() const override { return ASTKind::FOR_LOOP; }
    void print(int indent) const override;
    std::any accept(Visitor& visitor) override;
};

struct IfNode : ASTNode {
    std::shared_ptr<ASTNode> condition;
    std::vector<std::shared_ptr<ASTNode>> body;
    std::vector<std::shared_ptr<ASTNode>> orelse;
    ASTKind kind() const override { return ASTKind::IF; }
    void print(int indent) const override;
    std::any accept(Visitor& visitor) override;
};

struct AssignNode : ASTNode {
    std::shared_ptr<ASTNode> target;
    std::string op;
    std::shared_ptr<ASTNode> value;
    ASTKind kind() const override { return ASTKind::ASSIGN; }
    void print(int indent) const override;
    std::any accept(Visitor& visitor) override;
};

struct TensorDeclNode : ASTNode {
    std::string name;
    std::string dtype;
    std::vector<ShapeDim> shape;
    std::shared_ptr<ASTNode> init_value;
    ASTKind kind() const override { return ASTKind::TENSOR_DECL; }
    void print(int indent) const override;
    std::any accept(Visitor& visitor) override;
};

struct BinaryOpNode : ASTNode {
    std::shared_ptr<ASTNode> left;
    std::string op;
    std::shared_ptr<ASTNode> right;
    ASTKind kind() const override { return ASTKind::BINARY_OP; }
    void print(int indent) const override;
    std::any accept(Visitor& visitor) override;
};

struct UnaryOpNode : ASTNode {
    std::string op;
    std::shared_ptr<ASTNode> operand;
    ASTKind kind() const override { return ASTKind::UNARY_OP; }
    void print(int indent) const override;
    std::any accept(Visitor& visitor) override;
};

struct SubscriptNode : ASTNode {
    std::string value;
    std::vector<std::shared_ptr<ASTNode>> indices;
    ASTKind kind() const override { return ASTKind::SUBSCRIPT; }
    void print(int indent) const override;
    std::any accept(Visitor& visitor) override;
};

struct BoolOpNode : ASTNode {
    std::string op;
    std::vector<std::shared_ptr<ASTNode>> values;
    ASTKind kind() const override { return ASTKind::BOOL_OP; }
    void print(int indent) const override;
    std::any accept(Visitor& visitor) override;
};

struct TernaryNode : ASTNode {
    std::shared_ptr<ASTNode> cond;
    std::shared_ptr<ASTNode> body;
    std::shared_ptr<ASTNode> orelse;
    ASTKind kind() const override { return ASTKind::TERNARY; }
    void print(int indent) const override;
    std::any accept(Visitor& visitor) override;
};

struct VariableNode : ASTNode {
    std::string name;
    ASTKind kind() const override { return ASTKind::VARIABLE; }
    void print(int indent) const override;
    std::any accept(Visitor& visitor) override;
};

struct ConstantNode : ASTNode {
    double value;
    bool is_integer;
    ASTKind kind() const override { return ASTKind::CONSTANT; }
    void print(int indent) const override;
    std::any accept(Visitor& visitor) override;
};

struct ReturnNode : ASTNode {
    std::shared_ptr<ASTNode> value;
    ASTKind kind() const override { return ASTKind::RETURN; }
    void print(int indent) const override;
    std::any accept(Visitor& visitor) override;
};

struct BreakNode : ASTNode {
    ASTKind kind() const override { return ASTKind::BREAK; }
    void print(int indent) const override;
    std::any accept(Visitor& visitor) override;
};

struct ContinueNode : ASTNode {
    ASTKind kind() const override { return ASTKind::CONTINUE; }
    void print(int indent) const override;
    std::any accept(Visitor& visitor) override;
};

struct ListNode : ASTNode {
    std::vector<std::shared_ptr<ASTNode>> elements;
    ASTKind kind() const override { return ASTKind::LIST; }
    void print(int indent) const override;
    std::any accept(Visitor& visitor) override;
};

struct BuiltinCallNode : ASTNode {
    ASTKind kind() const override { return ASTKind::CUSTOM_OP; }
    std::string op;
    std::vector<std::shared_ptr<ASTNode>> args;
    void print(int indent) const override;
    std::any accept(Visitor& visitor) override;
};

struct PrintNode : ASTNode {
    std::shared_ptr<ASTNode> value;
    ASTKind kind() const override { return ASTKind::PRINT; }
    void print(int indent) const override;
    std::any accept(Visitor& visitor) override;
};

struct FunctionCallNode : ASTNode {
    std::string name;
    std::vector<std::shared_ptr<ASTNode>> args;
    ASTKind kind() const override { return ASTKind::FUNC_CALL; }
    void print(int indent) const override;
    std::any accept(Visitor& visitor) override;
};

class Visitor {
public:
    virtual ~Visitor() = default;
    virtual std::any visitFuncNode(std::shared_ptr<FuncNode> node);
    virtual std::any visitParamNode(std::shared_ptr<ParamNode> node);
    virtual std::any visitForLoopNode(std::shared_ptr<ForLoopNode> node);
    virtual std::any visitIfNode(std::shared_ptr<IfNode> node);
    virtual std::any visitAssignNode(std::shared_ptr<AssignNode> node);
    virtual std::any visitTensorDeclNode(std::shared_ptr<TensorDeclNode> node);
    virtual std::any visitBinaryOpNode(std::shared_ptr<BinaryOpNode> node);
    virtual std::any visitUnaryOpNode(std::shared_ptr<UnaryOpNode> node);
    virtual std::any visitSubscriptNode(std::shared_ptr<SubscriptNode> node);
    virtual std::any visitBoolOpNode(std::shared_ptr<BoolOpNode> node);
    virtual std::any visitTernaryNode(std::shared_ptr<TernaryNode> node);
    virtual std::any visitVariableNode(std::shared_ptr<VariableNode> node);
    virtual std::any visitConstantNode(std::shared_ptr<ConstantNode> node);
    virtual std::any visitReturnNode(std::shared_ptr<ReturnNode> node);
    virtual std::any visitBreakNode(std::shared_ptr<BreakNode> node);
    virtual std::any visitContinueNode(std::shared_ptr<ContinueNode> node);
    virtual std::any visitListNode(std::shared_ptr<ListNode> node);
    virtual std::any visitBuiltinCallNode(std::shared_ptr<BuiltinCallNode> node);
    virtual std::any visitPrintNode(std::shared_ptr<PrintNode> node);
    virtual std::any visitFunctionCallNode(std::shared_ptr<FunctionCallNode> node);
};

std::string indent_str(int depth);

#endif
