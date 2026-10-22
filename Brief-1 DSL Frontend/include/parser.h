#ifndef TLDR_PARSER_H
#define TLDR_PARSER_H

#include "tokens.h"
#include "ast_nodes.h"
#include "builtin_ops.h"
#include <vector>
#include <stdexcept>
#include <unordered_map>

class ParseError : public std::runtime_error {
public:
    int line, column;
    ParseError(const std::string& msg, int l, int c)
        : std::runtime_error("Parse error at L" + std::to_string(l) + ":" + std::to_string(c) + ": " + msg),
          line(l), column(c) {}
};

class Parser {
public:
    explicit Parser(const std::vector<Token>& tokens);
    std::vector<std::shared_ptr<FuncNode>> parse();

private:
    std::vector<Token> tokens_;
    size_t pos_;

    const Token& peek() const;
    const Token& advance();
    const Token& expect(TokenType type);
    const Token* match(TokenType type);
    void skip_newlines();
    bool at(TokenType type) const;
    bool at_any(std::initializer_list<TokenType> types) const;

    std::shared_ptr<FuncNode> parse_func_def();
    std::vector<std::shared_ptr<ParamNode>> parse_param_list();
    std::shared_ptr<ParamNode> parse_param();
    std::pair<std::string, std::vector<ShapeDim>> parse_type_anno();
    std::string parse_dtype();
    std::vector<ShapeDim> parse_shape();
    std::vector<std::shared_ptr<ASTNode>> parse_block();
    std::shared_ptr<ASTNode> parse_stmt();
    std::shared_ptr<ForLoopNode> parse_for_stmt();
    std::shared_ptr<IfNode> parse_if_stmt();
    std::shared_ptr<ReturnNode> parse_return_stmt();
    std::shared_ptr<ASTNode> parse_assign_or_tensor_decl();
    std::shared_ptr<ASTNode> parse_expr();
    std::shared_ptr<ASTNode> parse_or_expr();
    std::shared_ptr<ASTNode> parse_and_expr();
    std::shared_ptr<ASTNode> parse_comp_expr();
    std::shared_ptr<ASTNode> parse_add_expr();
    std::shared_ptr<ASTNode> parse_mul_expr();
    std::shared_ptr<ASTNode> parse_unary_expr();
    std::shared_ptr<ASTNode> parse_primary();
    std::shared_ptr<ASTNode> parse_builtin_call(const BuiltinOpSpec& spec, const Token& name_tok);
    std::vector<std::shared_ptr<ASTNode>> parse_expr_list();
    std::string expr_to_bound(const std::shared_ptr<ASTNode>& expr);
};

#endif
