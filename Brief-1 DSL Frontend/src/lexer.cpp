#include "lexer.h"
#include <cctype>
#include <unordered_map>

//essentially a dictionary that will return the Token as soon as it is matched.
//only implemebted for Keywords, nothing more.
//Builtin operators (relu, softmax, ...) are deliberately NOT keywords here:
//they are recognised by the parser through the registry in include/builtin_ops.h.
static const std::unordered_map<std::string, TokenType> KEYWORDS = {
    {"return", TokenType::TOK_RETURN},
    {"for",    TokenType::TOK_FOR},
    {"in",     TokenType::TOK_IN},
    {"loop",   TokenType::TOK_LOOP},
    {"if",     TokenType::TOK_IF},
    {"else",   TokenType::TOK_ELSE},
    {"and",    TokenType::TOK_AND},
    {"or",     TokenType::TOK_OR},
    {"break",  TokenType::TOK_BREAK},
    {"continue",TokenType::TOK_CONTINUE},
    {"def",    TokenType::TOK_DEF},
    {"print",  TokenType::TOK_PRINT},
    {"Tensor", TokenType::TOK_TENSOR},
    {"f32",    TokenType::TOK_F32},
    {"f64",    TokenType::TOK_F64},
    {"i32",    TokenType::TOK_I32},
};

//This has been implemented for taking the string buffer from the code file.
Lexer::Lexer(const std::string& source)
    : source_(source), pos_(0), line_(1), column_(1) {}

//peek() is used to only look at the character, not consume it
char Lexer::peek() const {
    if (pos_ < source_.size()) return source_[pos_];
    return '\0';
}
//peek_next() is used to look one character ahead. It also will not consume the character
char Lexer::peek_next() const {
    if (pos_ + 1 < source_.size()) return source_[pos_ + 1];
    return '\0';
}
//advance() will consume and give current character and move onto the next 
char Lexer::advance() {
    char ch = source_[pos_++]; //source will give pos_, and then pos_ will increment. Called postfix increment
    if (ch == '\n') { line_++; column_ = 1; }
    else { column_++; }
    return ch;
}
//not used in code, but implemented a base version for more complex regex and pattern matching if needed.
bool Lexer::match(char expected) {
    if (pos_ < source_.size() && source_[pos_] == expected) {
        advance();
        return true;
    }
    return false;
}
//self explanatory
void Lexer::skip_whitespace_and_comments() {
    while (pos_ < source_.size()) {
        char ch = source_[pos_];
        if (ch == ' ' || ch == '\t' || ch == '\r') {
            advance();
        } else if (ch == '#') {
            while (pos_ < source_.size() && source_[pos_] != '\n') advance();
        } else {
            break;
        }
    }
}
//helper function to read floats or integers
Token Lexer::read_number() {
    int start_line = line_, start_col = column_;
    std::string text;
    bool has_dot = false;
    while (pos_ < source_.size() && (std::isdigit(source_[pos_]) || (source_[pos_] == '.' && !has_dot))) {
        if (source_[pos_] == '.') has_dot = true;
        text += advance();
    }
    if (has_dot)
        return Token(TokenType::TOK_FLOAT, text, start_line, start_col);
    return Token(TokenType::TOK_INTEGER, text, start_line, start_col);
}

Token Lexer::read_identifier() {
    int start_line = line_, start_col = column_;
    std::string text;
    //std::isalnum() is used to check if a character is alphanumeric
    while (pos_ < source_.size() && (std::isalnum(source_[pos_]) || source_[pos_] == '_')) {
        text += advance();
    }
    auto it = KEYWORDS.find(text); //the map will return a .end() if the key is not found. It is what we are checking below.
    TokenType type = (it != KEYWORDS.end()) ? it->second : TokenType::TOK_IDENT;
    return Token(type, text, start_line, start_col);
}

std::vector<Token> Lexer::tokenize() {
    while (pos_ < source_.size()) {
        skip_whitespace_and_comments();
        if (pos_ >= source_.size()) break;

        char ch = peek();
        int start_line = line_, start_col = column_;

        // Newlines are significant for block structure
        if (ch == '\n') {
            advance();
            continue;
        }

        // Simple checking for Two-character operators
        // We dont need super complex regex patter matching, since have strictly defined out DSL
        if (ch == '-' && peek_next() == '>') {
            advance(); advance();
            tokens_.emplace_back(TokenType::TOK_ARROW, "->", start_line, start_col);
        } else if (ch == '+' && peek_next() == '=') {
            advance(); advance();
            tokens_.emplace_back(TokenType::TOK_PLUS_EQ, "+=", start_line, start_col);
        } else if (ch == '-' && peek_next() == '=') {
            advance(); advance();
            tokens_.emplace_back(TokenType::TOK_MINUS_EQ, "-=", start_line, start_col);
        } else if (ch == '*' && peek_next() == '=') {
            advance(); advance();
            tokens_.emplace_back(TokenType::TOK_STAR_EQ, "*=", start_line, start_col);
        } else if (ch == '/' && peek_next() == '=') {
            advance(); advance();
            tokens_.emplace_back(TokenType::TOK_SLASH_EQ, "/=", start_line, start_col);
        } else if (ch == '=' && peek_next() == '=') {
            advance(); advance();
            tokens_.emplace_back(TokenType::TOK_EQ_EQ, "==", start_line, start_col);
        } else if (ch == '!' && peek_next() == '=') {
            advance(); advance();
            tokens_.emplace_back(TokenType::TOK_NOT_EQ, "!=", start_line, start_col);
        } else if (ch == '<' && peek_next() == '=') {
            advance(); advance();
            tokens_.emplace_back(TokenType::TOK_LE, "<=", start_line, start_col);
        } else if (ch == '>' && peek_next() == '=') {
            advance(); advance();
            tokens_.emplace_back(TokenType::TOK_GE, ">=", start_line, start_col);
        }
        // Single-character tokens
        else if (ch == '+') { advance(); tokens_.emplace_back(TokenType::TOK_PLUS, "+", start_line, start_col); }
        else if (ch == '-') { advance(); tokens_.emplace_back(TokenType::TOK_MINUS, "-", start_line, start_col); }
        else if (ch == '*') { advance(); tokens_.emplace_back(TokenType::TOK_STAR, "*", start_line, start_col); }
        else if (ch == '/') { advance(); tokens_.emplace_back(TokenType::TOK_SLASH, "/", start_line, start_col); }
        else if (ch == '=') { advance(); tokens_.emplace_back(TokenType::TOK_EQ, "=", start_line, start_col); }
        else if (ch == '<') { advance(); tokens_.emplace_back(TokenType::TOK_LT, "<", start_line, start_col); }
        else if (ch == '>') { advance(); tokens_.emplace_back(TokenType::TOK_GT, ">", start_line, start_col); }
        else if (ch == '(') { advance(); tokens_.emplace_back(TokenType::TOK_LPAREN, "(", start_line, start_col); }
        else if (ch == ')') { advance(); tokens_.emplace_back(TokenType::TOK_RPAREN, ")", start_line, start_col); }
        else if (ch == '[') { advance(); tokens_.emplace_back(TokenType::TOK_LBRACKET, "[", start_line, start_col); }
        else if (ch == ']') { advance(); tokens_.emplace_back(TokenType::TOK_RBRACKET, "]", start_line, start_col); }
        else if (ch == '{') { advance(); tokens_.emplace_back(TokenType::TOK_LBRACE, "{", start_line, start_col); }
        else if (ch == '}') { advance(); tokens_.emplace_back(TokenType::TOK_RBRACE, "}", start_line, start_col); }
        else if (ch == ',') { advance(); tokens_.emplace_back(TokenType::TOK_COMMA, ",", start_line, start_col); }
        else if (ch == ':') { advance(); tokens_.emplace_back(TokenType::TOK_COLON, ":", start_line, start_col); }
        else if (ch == ';') { advance(); tokens_.emplace_back(TokenType::TOK_SEMICOLON, ";", start_line, start_col); }
        // Numbers and identifiers are being handled in these 2 else if statements
        else if (std::isdigit(ch)) { tokens_.push_back(read_number()); }
        else if (std::isalpha(ch) || ch == '_') { 
            tokens_.push_back(read_identifier());    
        }
        else {
            throw LexerError(std::string("Unexpected character '") + ch + "'", start_line, start_col);
        }
    }
    tokens_.emplace_back(TokenType::TOK_EOF, "", line_, column_);
    return tokens_;
}
