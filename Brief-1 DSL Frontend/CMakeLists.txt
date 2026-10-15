cmake_minimum_required(VERSION 3.14)
project(tldr_compiler LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(tldr_compiler
    src/main.cpp
    src/lexer.cpp
    src/parser.cpp
    src/ast_nodes.cpp
    src/tokens.cpp
    src/json_visitor.cpp
    src/semantic.cpp
)

target_include_directories(tldr_compiler PRIVATE include)
