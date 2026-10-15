#ifndef TLDR_TOKENS_H
#define TLDR_TOKENS_H

#include <string>
#include <cstdint>

enum class TokenType {
    // Keywords
    TOK_RETURN,
    TOK_FOR,
    TOK_IN,
    TOK_LOOP,
    TOK_IF,
    TOK_ELSE,
    TOK_AND,
    TOK_OR,
    TOK_BREAK,
    TOK_CONTINUE,
    TOK_DEF,
    TOK_PRINT,

    // Type keywords
    TOK_TENSOR,
    TOK_F32,
    TOK_F64,
    TOK_I32,

    // Operators
    TOK_PLUS,
    TOK_MINUS,
    TOK_STAR,
    TOK_SLASH,
    TOK_EQ,
    TOK_PLUS_EQ,
    TOK_MINUS_EQ,
    TOK_STAR_EQ,
    TOK_SLASH_EQ,
    TOK_LT,
    TOK_GT,
    TOK_LE,
    TOK_GE,
    TOK_EQ_EQ,
    TOK_NOT_EQ,

    // Delimiters
    TOK_LPAREN,
    TOK_RPAREN,
    TOK_LBRACKET,
    TOK_RBRACKET,
    TOK_LBRACE,
    TOK_RBRACE,
    TOK_COMMA,
    TOK_COLON,
    TOK_SEMICOLON,
    TOK_ARROW,

    // Literals
    TOK_INTEGER,
    TOK_FLOAT,
    TOK_IDENT,

    // Special
    TOK_EOF
};

struct Token {
    TokenType type;
    std::string value;
    int line;
    int column;

    Token() : type(TokenType::TOK_EOF), line(0), column(0) {}
    Token(TokenType t, const std::string& v, int l, int c)
        : type(t), value(v), line(l), column(c) {}
};

const char* token_type_name(TokenType t);

#endif
