#include "parser.h"
#include <sstream>

Parser::Parser(const std::vector<Token>& tokens) : tokens_(tokens), pos_(0) {}

const Token& Parser::peek() const { return tokens_[pos_]; }

const Token& Parser::advance() {
    if (pos_ < tokens_.size() - 1) pos_++;
    return tokens_[pos_ - 1]; 
}

const Token& Parser::expect(TokenType type) {
    const Token& tok = peek();
    if (tok.type != type) {
        throw ParseError("Expected " + std::string(token_type_name(type)) +
                         ", got " + std::string(token_type_name(tok.type)) +
                         " (" + tok.value + ")", tok.line, tok.column);
    }
    return advance();
}

const Token* Parser::match(TokenType type) {
    if (peek().type == type) { return &advance(); }
    return nullptr;
}

void Parser::skip_newlines() {
    //lexer strips newlines so the token stream never has them.
    //declaring just to not cause errors
}

bool Parser::at(TokenType type) const { return peek().type == type; }

bool Parser::at_any(std::initializer_list<TokenType> types) const {
    for (auto t : types) if (peek().type == t) return true;
    return false;
}

// The lexer skips whitespace and comments using skip_whitespace_and_comments() ,
// So newlines ARE consumed but no NEWLINE token is produced.
// No need to implement it here

std::vector<std::shared_ptr<FuncNode>> Parser::parse() {
    std::vector<std::shared_ptr<FuncNode>> funcs; //structured because at the top level we expect functions
    //this will be changed in the future. We want to make it a more natural and diverse language.
    while (!at(TokenType::TOK_EOF)) {
        funcs.push_back(parse_func_def());
    }
    return funcs;
}

std::shared_ptr<FuncNode> Parser::parse_func_def() {
    auto node = std::make_shared<FuncNode>();
    // Function names are plain identifiers, so nothing besides TOK_IDENT has
    // to be accepted here (builtin ops are resolved later through the registry).
    if (!at(TokenType::TOK_IDENT)) {
        throw ParseError("Expected function name, got " +
                         std::string(token_type_name(peek().type)) + " (" + peek().value + ")",
                         peek().line, peek().column);
    }
    Token tok = advance();
    node->name = tok.value; //.value is the member variable, of the token returned by expect
    node->sp.line = tok.line;
    node->sp.col = tok.column;
    expect(TokenType::TOK_LPAREN);
    node->params = parse_param_list();
    expect(TokenType::TOK_RPAREN);
    //simple use of match function to check if syntax is properly used
    if (match(TokenType::TOK_ARROW)) {
        expect(TokenType::TOK_LBRACKET);
        if (!at(TokenType::TOK_RBRACKET)) {
            node->returns.push_back(advance().value);
            while (match(TokenType::TOK_COMMA)) {
                node->returns.push_back(advance().value);
            }
        }
        expect(TokenType::TOK_RBRACKET);
    }

    expect(TokenType::TOK_LBRACE);
    node->body = parse_block();
    expect(TokenType::TOK_RBRACE);
    return node;
}

std::vector<std::shared_ptr<ParamNode>> Parser::parse_param_list() {
    std::vector<std::shared_ptr<ParamNode>> params;
    if (at(TokenType::TOK_RPAREN)) return params;
    params.push_back(parse_param());
    while (match(TokenType::TOK_COMMA)) {
        params.push_back(parse_param());
    }
    return params;
}

std::shared_ptr<ASTNode> Parser::parse_builtin_call(const BuiltinOpSpec& spec, const Token& name_tok) {
    expect(TokenType::TOK_LPAREN);

    auto node = std::make_shared<BuiltinCallNode>();
    node->op = spec.name;
    node->sp.line = name_tok.line;
    node->sp.col = name_tok.column;

    if (!at(TokenType::TOK_RPAREN)) {
        node->args.push_back(parse_expr());
        while (match(TokenType::TOK_COMMA)) {
            if (at(TokenType::TOK_RPAREN)) break;
            node->args.push_back(parse_expr());
        }
    }
    expect(TokenType::TOK_RPAREN);

    if (node->args.size() != spec.arity) {
        throw ParseError(
            "'" + spec.name + "' expects " + std::to_string(spec.arity) +
            " argument(s), got " + std::to_string(node->args.size()),
            name_tok.line, name_tok.column
        );
    }
    return node;
}

std::shared_ptr<ParamNode> Parser::parse_param() {
    auto node = std::make_shared<ParamNode>();
    Token tok = expect(TokenType::TOK_IDENT);
    node->name = tok.value; //.value is the member variable, of the token returned by expect
    node->sp.line = tok.line;
    node->sp.col = tok.column;
    expect(TokenType::TOK_COLON);
    auto [dtype, shape] = parse_type_anno();
    node->dtype = dtype;
    node->shape = shape;
    return node;
}

std::string Parser::parse_dtype() {
    Token tok = advance();
    if (tok.type != TokenType::TOK_F32 && tok.type != TokenType::TOK_F64 &&
        tok.type != TokenType::TOK_I32) {
        throw ParseError(
            "invalid data type '" + tok.value + "'; expected f32, f64 or i32",
            tok.line, tok.column
        );
    }
    return tok.value;
}

std::pair<std::string, std::vector<ShapeDim>> Parser::parse_type_anno() {
    if (match(TokenType::TOK_TENSOR)) {
        expect(TokenType::TOK_LBRACKET);
        std::string dtype = parse_dtype();
        expect(TokenType::TOK_COMMA);
        auto shape = parse_shape();
        expect(TokenType::TOK_RBRACKET);
        return {dtype, shape};
    } else {
        std::string dtype = parse_dtype();
        return {dtype, {}};
    }
}

std::vector<ShapeDim> Parser::parse_shape() {
    expect(TokenType::TOK_LPAREN);
    std::vector<ShapeDim> dims;
    if (!at(TokenType::TOK_RPAREN)) {
        // parse first dim: could be int literal or identifier
        if (at(TokenType::TOK_INTEGER)) {
            dims.push_back(std::stoll(advance().value));
        } else {
            dims.push_back(advance().value);
        }
        while (match(TokenType::TOK_COMMA)) {
            if (at(TokenType::TOK_RPAREN)) break; // trailing comma
            if (at(TokenType::TOK_INTEGER)) {
                dims.push_back(std::stoll(advance().value));
            } else {
                dims.push_back(advance().value);
            }
        }
    }
    expect(TokenType::TOK_RPAREN);
    return dims;
}

std::vector<std::shared_ptr<ASTNode>> Parser::parse_block() {
    std::vector<std::shared_ptr<ASTNode>> stmts;
    while (!at(TokenType::TOK_RBRACE) && !at(TokenType::TOK_EOF)) {
        stmts.push_back(parse_stmt());
    }
    return stmts;
}
//function we use to call approprotiate parse function.
std::shared_ptr<ASTNode> Parser::parse_stmt() {
    if (at(TokenType::TOK_FOR)) return parse_for_stmt();
    if (at(TokenType::TOK_IF)) return parse_if_stmt();
    if (at(TokenType::TOK_RETURN)) return parse_return_stmt();
    if (at(TokenType::TOK_PRINT)) {
        const Token& tok = advance();
        auto node = std::make_shared<PrintNode>();
        node->sp.line = tok.line;
        node->sp.col = tok.column;
        expect(TokenType::TOK_LPAREN);
        node->value = parse_expr();
        expect(TokenType::TOK_RPAREN);
        match(TokenType::TOK_SEMICOLON);
        return node;
    }
    if (at(TokenType::TOK_BREAK)) {
        const Token& tok = advance();
        match(TokenType::TOK_SEMICOLON);
        auto node = std::make_shared<BreakNode>();
        node->sp.line = tok.line;
        node->sp.col = tok.column;
        return node;
    }
    if (at(TokenType::TOK_CONTINUE)) {
        const Token& tok = advance();
        match(TokenType::TOK_SEMICOLON);
        auto node = std::make_shared<ContinueNode>();
        node->sp.line = tok.line;
        node->sp.col = tok.column;
        return node;
    }
    return parse_assign_or_tensor_decl();
}

std::shared_ptr<ForLoopNode> Parser::parse_for_stmt() {
    auto node = std::make_shared<ForLoopNode>();
    Token tok = expect(TokenType::TOK_FOR);
    node->sp.line = tok.line;
    node->sp.col = tok.column;
    node->var = expect(TokenType::TOK_IDENT).value;
    expect(TokenType::TOK_IN);
    expect(TokenType::TOK_LOOP);
    expect(TokenType::TOK_LPAREN);
    auto bound_expr = parse_expr();
    node->bound = expr_to_bound(bound_expr);
    while (match(TokenType::TOK_COMMA)) {
        node->attr_type = expect(TokenType::TOK_IDENT).value;
        expect(TokenType::TOK_EQ);
        node->tile_or_unroll = std::stoi(expect(TokenType::TOK_INTEGER).value);
    }
    expect(TokenType::TOK_RPAREN);
    expect(TokenType::TOK_LBRACE);
    node->body = parse_block();
    expect(TokenType::TOK_RBRACE);
    return node;
}

std::string Parser::expr_to_bound(const std::shared_ptr<ASTNode>& expr) {
    if (auto var = dynamic_cast<VariableNode*>(expr.get())) return var->name;
    if (auto c = dynamic_cast<ConstantNode*>(expr.get())) {
        if (c->is_integer) return std::to_string((int64_t)c->value);
        return std::to_string(c->value);
    }
    if (auto bin = dynamic_cast<BinaryOpNode*>(expr.get()))
        return expr_to_bound(bin->left) + " " + bin->op + " " + expr_to_bound(bin->right);
    return "?";
}

std::shared_ptr<IfNode> Parser::parse_if_stmt() {
    auto node = std::make_shared<IfNode>();
    Token tok = expect(TokenType::TOK_IF);
    node->sp.line = tok.line;
    node->sp.col = tok.column;
    node->condition = parse_expr();
    expect(TokenType::TOK_LBRACE);
    node->body = parse_block();
    expect(TokenType::TOK_RBRACE);
    if (match(TokenType::TOK_ELSE)) {
        expect(TokenType::TOK_LBRACE);
        node->orelse = parse_block();
        expect(TokenType::TOK_RBRACE);
    }
    return node;
}

std::shared_ptr<ReturnNode> Parser::parse_return_stmt() {
    auto node = std::make_shared<ReturnNode>();
    Token tok = expect(TokenType::TOK_RETURN);
    node->sp.line = tok.line;
    node->sp.col = tok.column;
    if (!at(TokenType::TOK_SEMICOLON) && !at(TokenType::TOK_RBRACE) && !at(TokenType::TOK_EOF)) {
        node->value = parse_expr();
    }
    match(TokenType::TOK_SEMICOLON);
    return node;
}

std::shared_ptr<ASTNode> Parser::parse_assign_or_tensor_decl() {
    Token tok = expect(TokenType::TOK_IDENT);

    // Tensor declaration: name : Tensor[dtype, (shape)]
    if (match(TokenType::TOK_COLON)) {
        auto node = std::make_shared<TensorDeclNode>();
        node->name = tok.value;
        node->sp.line = tok.line;
        node->sp.col = tok.column;
        auto [dtype, shape] = parse_type_anno();
        node->dtype = dtype;
        node->shape = shape;
        if (match(TokenType::TOK_EQ)) {
            node->init_value = parse_expr();
        }
        match(TokenType::TOK_SEMICOLON);
        return node;
    }

    // Could be subscript target: name[expr, ...] = expr
    std::shared_ptr<ASTNode> target;
    auto var = std::make_shared<VariableNode>();
    var->name = tok.value;
    var->sp.line = tok.line;
    var->sp.col = tok.column;
    target = var;

    if (match(TokenType::TOK_LBRACKET)) {
        auto sub = std::make_shared<SubscriptNode>();
        sub->value = tok.value;
        sub->sp.line = tok.line;
        sub->sp.col = tok.column;
        sub->indices = parse_expr_list();
        expect(TokenType::TOK_RBRACKET);
        target = sub;
    }

    // Assignment operator
    const Token& op_tok = peek();
    std::string op;
    if (op_tok.type == TokenType::TOK_EQ) op = "=";
    else if (op_tok.type == TokenType::TOK_PLUS_EQ) op = "+=";
    else if (op_tok.type == TokenType::TOK_MINUS_EQ) op = "-=";
    else if (op_tok.type == TokenType::TOK_STAR_EQ) op = "*=";
    else if (op_tok.type == TokenType::TOK_SLASH_EQ) op = "/=";
    else throw ParseError("Expected assignment operator, got " + std::string(token_type_name(op_tok.type)), op_tok.line, op_tok.column);
    advance();

    auto value = parse_expr();
    match(TokenType::TOK_SEMICOLON);

    auto node = std::make_shared<AssignNode>();
    node->target = target;
    node->sp.line = var->sp.line;
    node->sp.col = var->sp.col;
    node->op = op;
    node->value = value;
    return node;
}

std::shared_ptr<ASTNode> Parser::parse_expr() { return parse_or_expr(); }

std::shared_ptr<ASTNode> Parser::parse_or_expr() {
    auto left = parse_and_expr();
    while (at(TokenType::TOK_OR)) {
        const Token& op_tok = advance();
        auto right = parse_and_expr();
        auto node = std::make_shared<BoolOpNode>();
        node->op = "or";
        node->sp.line = op_tok.line;
        node->sp.col = op_tok.column;
        node->values.push_back(left);
        node->values.push_back(right);
        left = node;
    }
    return left;
}

std::shared_ptr<ASTNode> Parser::parse_and_expr() {
    auto left = parse_comp_expr();
    while (at(TokenType::TOK_AND)) {
        const Token& op_tok = advance();
        auto right = parse_comp_expr();
        auto node = std::make_shared<BoolOpNode>();
        node->op = "and";
        node->sp.line = op_tok.line;
        node->sp.col = op_tok.column;
        node->values.push_back(left);
        node->values.push_back(right);
        left = node;
    }
    return left;
}

std::shared_ptr<ASTNode> Parser::parse_comp_expr() {
    auto left = parse_add_expr();
    if (at_any({TokenType::TOK_LT, TokenType::TOK_GT, TokenType::TOK_LE,
                TokenType::TOK_GE, TokenType::TOK_EQ_EQ, TokenType::TOK_NOT_EQ})) {
        const Token& op_tok = advance();
        std::string op = op_tok.value;
        auto right = parse_add_expr();
        auto node = std::make_shared<BinaryOpNode>();
        node->left = left;
        node->op = op;
        node->sp.line = op_tok.line;
        node->sp.col = op_tok.column;
        node->right = right;
        return node;
    }
    return left;
}

std::shared_ptr<ASTNode> Parser::parse_add_expr() {
    auto left = parse_mul_expr();
    while (at(TokenType::TOK_PLUS) || at(TokenType::TOK_MINUS)) {
        const Token& op_tok = advance();
        std::string op = op_tok.value;
        auto right = parse_mul_expr();
        auto node = std::make_shared<BinaryOpNode>();
        node->left = left;
        node->op = op;
        node->sp.line = op_tok.line;
        node->sp.col = op_tok.column;
        node->right = right;
        left = node;
    }
    return left;
}

std::shared_ptr<ASTNode> Parser::parse_mul_expr() {
    auto left = parse_unary_expr();
    while (at(TokenType::TOK_STAR) || at(TokenType::TOK_SLASH)) {
        const Token& op_tok = advance();
        std::string op = op_tok.value;
        auto right = parse_unary_expr();
        auto node = std::make_shared<BinaryOpNode>();
        node->left = left;
        node->op = op;
        node->sp.line = op_tok.line;
        node->sp.col = op_tok.column;
        node->right = right;
        left = node;
    }
    return left;
}

std::shared_ptr<ASTNode> Parser::parse_unary_expr() {
    if (at(TokenType::TOK_MINUS)) {
        const Token& op_tok = advance();
        auto operand = parse_unary_expr();
        auto node = std::make_shared<UnaryOpNode>();
        node->op = "-";
        node->sp.line = op_tok.line;
        node->sp.col = op_tok.column;
        node->operand = operand;
        return node;
    }
    return parse_primary();
}

std::shared_ptr<ASTNode> Parser::parse_primary() {
    const Token& tok = peek();

    if (tok.type == TokenType::TOK_INTEGER) {
        advance();
        auto node = std::make_shared<ConstantNode>();
        node->value = std::stod(tok.value);
        node->is_integer = true;
        node->sp.line = tok.line;
        node->sp.col = tok.column;
        return node;
    }
    if (tok.type == TokenType::TOK_FLOAT) {
        advance();
        auto node = std::make_shared<ConstantNode>();
        node->value = std::stod(tok.value);
        node->is_integer = false;
        node->sp.line = tok.line;
        node->sp.col = tok.column;
        return node;
    }
    if (tok.type == TokenType::TOK_LPAREN) {
        advance();
        auto expr = parse_expr();
        expect(TokenType::TOK_RPAREN);
        return expr;
    }
    if (tok.type == TokenType::TOK_LBRACKET) {
        advance();
        auto node = std::make_shared<ListNode>();
        node->sp.line = tok.line;
        node->sp.col = tok.column;
        if (!at(TokenType::TOK_RBRACKET)) {
            node->elements.push_back(parse_expr());
            while (match(TokenType::TOK_COMMA)) {
                if (at(TokenType::TOK_RBRACKET)) break;
                node->elements.push_back(parse_expr());
            }
        }
        expect(TokenType::TOK_RBRACKET);
        return node;
    }
    if (tok.type == TokenType::TOK_IDENT) {
        const Token name_tok = advance();
        const std::string& name = name_tok.value;

        if (at(TokenType::TOK_LPAREN)) {
            // Builtin operator call (relu(...), softmax(...), ...) resolved
            // through the registry, so new operators need no parser change.
            if (const BuiltinOpSpec* spec = find_builtin_op(name)) {
                return parse_builtin_call(*spec, name_tok);
            }
            // User-defined function call
            advance();
            auto node = std::make_shared<FunctionCallNode>();
            node->name = name;
            node->sp.line = name_tok.line;
            node->sp.col = name_tok.column;
            if (!at(TokenType::TOK_RPAREN)) {
                node->args.push_back(parse_expr());
                while (match(TokenType::TOK_COMMA)) {
                    if (at(TokenType::TOK_RPAREN)) break;
                    node->args.push_back(parse_expr());
                }
            }
            expect(TokenType::TOK_RPAREN);
            return node;
        }

        // Builtin names are operators, not values: they are only valid in a call.
        if (find_builtin_op(name) != nullptr) {
            throw ParseError("'" + name + "' must be called with parentheses", name_tok.line, name_tok.column);
        }

        if (match(TokenType::TOK_LBRACKET)) {
            auto node = std::make_shared<SubscriptNode>();
            node->value = name;
            node->sp.line = name_tok.line;
            node->sp.col = name_tok.column;
            node->indices = parse_expr_list();
            expect(TokenType::TOK_RBRACKET);
            return node;
        }

        auto node = std::make_shared<VariableNode>();
        node->name = name;
        node->sp.line = name_tok.line;
        node->sp.col = name_tok.column;
        return node;
    }
    throw ParseError("Unexpected token " + std::string(token_type_name(tok.type)) + " (" + tok.value + ")", tok.line, tok.column);
}

std::vector<std::shared_ptr<ASTNode>> Parser::parse_expr_list() {
    auto list = std::vector<std::shared_ptr<ASTNode>>();
    list.push_back(parse_expr());
    while (match(TokenType::TOK_COMMA)) {
        if (at(TokenType::TOK_RBRACKET) || at(TokenType::TOK_RPAREN)) break;
        list.push_back(parse_expr());
    }
    return list;
}
