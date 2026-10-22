#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include "lexer.h"
#include "parser.h"
#include "json_visitor.h"
#include "graphviz_visitor.h"
#include "semantic.h"

std::string read_file(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + path);
    }
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

std::string strip_extension(const std::string& path) {
    size_t dot = path.rfind('.');
    if (dot != std::string::npos) return path.substr(0, dot);
    return path;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: tldr_compiler <file.tldr> [--tokens] [--json] [--dot]" << std::endl;
        return 1;
    }

    std::string path = argv[1];

    bool show_tokens = false;
    bool show_json = false;
    bool show_dot = false;

    for (int i = 2; i < argc; i++) {
        std::string flag = argv[i];
        if (flag == "--tokens") show_tokens = true;
        else if (flag == "--json") show_json = true;
        else if (flag == "--dot") show_dot = true;
    }

    try {
        std::string source = read_file(path);

        // Tokenize
        Lexer lexer(source);
        auto tokens = lexer.tokenize();

        // Print tokens (optional, use --tokens flag)
        if (show_tokens) {
            std::cout << "=== TOKENS ===" << std::endl;
            for (auto& tok : tokens) {
                std::cout << "  Token(" << token_type_name(tok.type)
                          << ", \"" << tok.value << "\""
                          << ", L" << tok.line << ":" << tok.column << ")" << std::endl;
            }
            std::cout << std::endl;
        }

        // Parse
        Parser parser(tokens);
        auto funcs = parser.parse();

        // Semantic analysis (symbol tables, scope/name resolution)
        SemanticAnalyzer analyzer;
        analyzer.analyze(funcs);

        // Print AST (always)
        std::cout << "=== AST ===" << std::endl;
        for (auto& func : funcs) {
            func->print(0);
            std::cout << std::endl;
        }

        // JSON output
        if (show_json) {
            JsonVisitor json_visitor;
            for (auto& func : funcs) {
                json j = std::any_cast<json>(func->accept(json_visitor));
                std::string json_path = strip_extension(path) + ".json";
                std::ofstream out(json_path);
                out << j.dump(4) << std::endl;
                out.close();
                std::cout << "JSON written to " << json_path << std::endl;
            }
        }

        // Graphviz DOT output
        if (show_dot) {
            GraphvizVisitor graphviz_visitor;
            for (auto& func : funcs) {
                std::string dot_path = strip_extension(path) + ".dot";
                graphviz_visitor.exportGraph(func, dot_path);
                std::cout << "DOT written to " << dot_path << std::endl;
            }
        }

    } catch (const LexerError& e) {
        std::cerr << "[Lexer Error] " << e.what() << std::endl;
        return 1;
    } catch (const ParseError& e) {
        std::cerr << "[Parse Error] " << e.what() << std::endl;
        return 1;
    } catch (const SemanticError& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "[Error] " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
