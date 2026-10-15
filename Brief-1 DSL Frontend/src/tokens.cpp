#include "tokens.h"
//used in case we need to inform user about what token is being talked about.
//useful because this can be used to tell user which token is causing an error, rather than returning an integer (since all tokens are enums)
const char* token_type_name(TokenType t) {
    switch (t) {
        case TokenType::TOK_RETURN:    return "RETURN";
        case TokenType::TOK_FOR:       return "FOR";
        case TokenType::TOK_IN:        return "IN";
        case TokenType::TOK_LOOP:      return "LOOP";
        case TokenType::TOK_IF:        return "IF";
        case TokenType::TOK_ELSE:      return "ELSE";
        case TokenType::TOK_AND:       return "AND";
        case TokenType::TOK_OR:        return "OR";
        case TokenType::TOK_BREAK:     return "BREAK";
        case TokenType::TOK_CONTINUE:  return "CONTINUE";
        case TokenType::TOK_TENSOR:    return "TENSOR";
        case TokenType::TOK_F32:       return "F32";
        case TokenType::TOK_F64:       return "F64";
        case TokenType::TOK_I32:       return "I32";
        case TokenType::TOK_PLUS:      return "PLUS";
        case TokenType::TOK_MINUS:     return "MINUS";
        case TokenType::TOK_STAR:      return "STAR";
        case TokenType::TOK_SLASH:     return "SLASH";
        case TokenType::TOK_EQ:        return "EQ";
        case TokenType::TOK_PLUS_EQ:   return "PLUS_EQ";
        case TokenType::TOK_MINUS_EQ:  return "MINUS_EQ";
        case TokenType::TOK_STAR_EQ:   return "STAR_EQ";
        case TokenType::TOK_SLASH_EQ:  return "SLASH_EQ";
        case TokenType::TOK_LT:        return "LT";
        case TokenType::TOK_GT:        return "GT";
        case TokenType::TOK_LE:        return "LE";
        case TokenType::TOK_GE:        return "GE";
        case TokenType::TOK_EQ_EQ:     return "EQ_EQ";
        case TokenType::TOK_NOT_EQ:    return "NOT_EQ";
        case TokenType::TOK_LPAREN:    return "LPAREN";
        case TokenType::TOK_RPAREN:    return "RPAREN";
        case TokenType::TOK_LBRACKET:  return "LBRACKET";
        case TokenType::TOK_RBRACKET:  return "RBRACKET";
        case TokenType::TOK_LBRACE:    return "LBRACE";
        case TokenType::TOK_RBRACE:    return "RBRACE";
        case TokenType::TOK_COMMA:     return "COMMA";
        case TokenType::TOK_COLON:     return "COLON";
        case TokenType::TOK_SEMICOLON: return "SEMICOLON";
        case TokenType::TOK_ARROW:     return "ARROW";
        case TokenType::TOK_INTEGER:   return "INTEGER";
        case TokenType::TOK_FLOAT:     return "FLOAT";
        case TokenType::TOK_IDENT:     return "IDENT";
        case TokenType::TOK_DEF:       return "DEF";
        case TokenType::TOK_PRINT:     return "PRINT";
        case TokenType::TOK_EOF:       return "EOF";
    }
    return "UNKNOWN";
}