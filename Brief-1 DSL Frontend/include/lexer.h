#ifndef TLDR_LEXER_H
#define TLDR_LEXER_H

#include "tokens.h"
#include <string>
#include <vector>
#include <stdexcept>

class LexerError : public std::runtime_error {
public:
    int line, column;
    LexerError(const std::string& msg, int l, int c)
        : std::runtime_error("Lexer error at L" + std::to_string(l) + ":" + std::to_string(c) + ": " + msg),
          line(l), column(c) {}
};

class Lexer {
public:
    explicit Lexer(const std::string& source);
    std::vector<Token> tokenize();

private:
    std::string source_;
    size_t pos_;
    int line_;
    int column_;
    std::vector<Token> tokens_;

    char peek() const;
    char peek_next() const;
    char advance();
    bool match(char expected);
    void skip_whitespace_and_comments();
    Token read_number();
    Token read_identifier();
};

#endif
