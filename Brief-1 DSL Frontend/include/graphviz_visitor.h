#ifndef TLDR_GRAPHVIZ_VISITOR_H
#define TLDR_GRAPHVIZ_VISITOR_H

#include "ast_nodes.h"
#include <fstream>
#include <string>
#include <any>
#include <sstream>

class GraphvizVisitor : public Visitor {
private:
    std::ofstream out;
    int nodeCounter = 0;

    int emitNode(const std::string& label, const std::string& color = "lightblue") {
        int id = nodeCounter++;
        out << "    node" << id << " [label=\"" << label
            << "\", shape=box, style=filled, fillcolor=" << color << "];\n";
        return id;
    }

    void emitEdge(int parentId, int childId) {
        if (childId != -1) {
            out << "    node" << parentId << " -> node" << childId << ";\n";
        }
    }

public:
    void exportGraph(const std::shared_ptr<ASTNode>& root, const std::string& filename) {
        out.open(filename);
        out << "digraph AST {\n";
        out << "    node [fontname=\"Helvetica\"];\n";

        nodeCounter = 0;
        if (root) {
            root->accept(*this);
        }

        out << "}\n";
        out.close();
    }

    std::any visitFuncNode(std::shared_ptr<FuncNode> node) override {
        std::string label = "Func: " + node->name;
        if (!node->returns.empty()) {
            label += "\\nreturns: ";
            for (size_t i = 0; i < node->returns.size(); i++) {
                if (i > 0) label += ", ";
                label += node->returns[i];
            }
        }
        int myId = emitNode(label, "lightyellow");

        for (auto& param : node->params) {
            auto paramId = std::any_cast<int>(param->accept(*this));
            emitEdge(myId, paramId);
        }

        for (const auto& stmt : node->body) {
            if (stmt) {
                auto stmtId = std::any_cast<int>(stmt->accept(*this));
                emitEdge(myId, stmtId);
            }
        }
        return myId;
    }

    std::any visitParamNode(std::shared_ptr<ParamNode> node) override {
        std::string shape_str;
        if (!node->shape.empty()) {
            shape_str = " [";
            for (size_t i = 0; i < node->shape.size(); i++) {
                if (i > 0) shape_str += ", ";
                std::visit([&shape_str](auto&& val) {
                    using T = std::decay_t<decltype(val)>;
                    if constexpr (std::is_same_v<T, std::string>) shape_str += val;
                    else shape_str += std::to_string(val);
                }, node->shape[i]);
            }
            shape_str += "]";
        }
        std::string label = "Param: " + node->name + " (" + node->dtype + shape_str + ")";
        return emitNode(label, "lightcyan");
    }

    std::any visitForLoopNode(std::shared_ptr<ForLoopNode> node) override {
        std::string label = "For (" + node->var + " in " + node->bound + ")\\n"
                          + node->attr_type + "=" + std::to_string(node->tile_or_unroll);
        int myId = emitNode(label, "orange");

        for (const auto& stmt : node->body) {
            if (stmt) {
                auto stmtId = std::any_cast<int>(stmt->accept(*this));
                emitEdge(myId, stmtId);
            }
        }
        return myId;
    }

    std::any visitIfNode(std::shared_ptr<IfNode> node) override {
        int myId = emitNode("If", "lightyellow");

        if (node->condition) {
            auto condId = std::any_cast<int>(node->condition->accept(*this));
            emitEdge(myId, condId);
        }

        for (const auto& stmt : node->body) {
            if (stmt) {
                auto stmtId = std::any_cast<int>(stmt->accept(*this));
                emitEdge(myId, stmtId);
            }
        }

        for (const auto& stmt : node->orelse) {
            if (stmt) {
                auto stmtId = std::any_cast<int>(stmt->accept(*this));
                emitEdge(myId, stmtId);
            }
        }
        return myId;
    }

    std::any visitAssignNode(std::shared_ptr<AssignNode> node) override {
        int myId = emitNode("Assign: " + node->op, "yellow");

        if (node->target) {
            auto targetId = std::any_cast<int>(node->target->accept(*this));
            emitEdge(myId, targetId);
        }
        if (node->value) {
            auto valueId = std::any_cast<int>(node->value->accept(*this));
            emitEdge(myId, valueId);
        }
        return myId;
    }

    std::any visitTensorDeclNode(std::shared_ptr<TensorDeclNode> node) override {
        std::string label = "Tensor: " + node->name + " (" + node->dtype + ")";
        return emitNode(label, "pink");
    }

    std::any visitBinaryOpNode(std::shared_ptr<BinaryOpNode> node) override {
        int myId = emitNode("BinOp: " + node->op, "lightyellow");

        if (node->left) {
            auto leftId = std::any_cast<int>(node->left->accept(*this));
            emitEdge(myId, leftId);
        }
        if (node->right) {
            auto rightId = std::any_cast<int>(node->right->accept(*this));
            emitEdge(myId, rightId);
        }
        return myId;
    }

    std::any visitUnaryOpNode(std::shared_ptr<UnaryOpNode> node) override {
        int myId = emitNode("UnaryOp: " + node->op, "lightyellow");

        if (node->operand) {
            auto operandId = std::any_cast<int>(node->operand->accept(*this));
            emitEdge(myId, operandId);
        }
        return myId;
    }

    std::any visitSubscriptNode(std::shared_ptr<SubscriptNode> node) override {
        int myId = emitNode("Subscript: " + node->value, "lavender");

        for (const auto& idx : node->indices) {
            if (idx) {
                auto idxId = std::any_cast<int>(idx->accept(*this));
                emitEdge(myId, idxId);
            }
        }
        return myId;
    }

    std::any visitBoolOpNode(std::shared_ptr<BoolOpNode> node) override {
        int myId = emitNode("BoolOp: " + node->op, "lightyellow");

        for (const auto& val : node->values) {
            if (val) {
                auto valId = std::any_cast<int>(val->accept(*this));
                emitEdge(myId, valId);
            }
        }
        return myId;
    }

    std::any visitTernaryNode(std::shared_ptr<TernaryNode> node) override {
        int myId = emitNode("Ternary", "lightyellow");

        if (node->cond) {
            auto condId = std::any_cast<int>(node->cond->accept(*this));
            emitEdge(myId, condId);
        }
        if (node->body) {
            auto bodyId = std::any_cast<int>(node->body->accept(*this));
            emitEdge(myId, bodyId);
        }
        if (node->orelse) {
            auto orelseId = std::any_cast<int>(node->orelse->accept(*this));
            emitEdge(myId, orelseId);
        }
        return myId;
    }

    std::any visitVariableNode(std::shared_ptr<VariableNode> node) override {
        return emitNode("Var: " + node->name, "lightgreen");
    }

    std::any visitConstantNode(std::shared_ptr<ConstantNode> node) override {
        return emitNode("Const: " + std::to_string(node->value), "gray");
    }

    std::any visitReturnNode(std::shared_ptr<ReturnNode> node) override {
        int myId = emitNode("Return", "lightyellow");

        if (node->value) {
            auto valId = std::any_cast<int>(node->value->accept(*this));
            emitEdge(myId, valId);
        }
        return myId;
    }

    std::any visitBreakNode(std::shared_ptr<BreakNode>) override {
        return emitNode("Break", "tomato");
    }

    std::any visitContinueNode(std::shared_ptr<ContinueNode>) override {
        return emitNode("Continue", "tomato");
    }

    std::any visitListNode(std::shared_ptr<ListNode> node) override {
        int myId = emitNode("List", "lightyellow");

        for (const auto& elem : node->elements) {
            if (elem) {
                auto elemId = std::any_cast<int>(elem->accept(*this));
                emitEdge(myId, elemId);
            }
        }
        return myId;
    }
    std::any visitBuiltinCallNode(std::shared_ptr<BuiltinCallNode> node) override {
        int myId = emitNode(node->op, "lightblue");

        for (const auto& arg : node->args) {
            if (arg) {
                auto argId = std::any_cast<int>(arg->accept(*this));
                emitEdge(myId, argId);
            }
        }
    return myId;
    }

    std::any visitPrintNode(std::shared_ptr<PrintNode> node) override {
        int myId = emitNode("Print", "lightyellow");

        if (node->value) {
            auto valId = std::any_cast<int>(node->value->accept(*this));
            emitEdge(myId, valId);
        }
        return myId;
    }

    std::any visitFunctionCallNode(std::shared_ptr<FunctionCallNode> node) override {
        int myId = emitNode("Call: " + node->name, "lightblue");

        for (const auto& arg : node->args) {
            if (arg) {
                auto argId = std::any_cast<int>(arg->accept(*this));
                emitEdge(myId, argId);
            }
        }
        return myId;
    }
};

#endif
