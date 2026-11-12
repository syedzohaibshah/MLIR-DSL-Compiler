#ifndef TLDR_BUILTIN_OPS_H
#define TLDR_BUILTIN_OPS_H

#include <string>
#include <unordered_map>
#include <vector>

// Single source of truth for every builtin operator of the DSL.
// To add a new operator, add one entry to builtin_ops() below.
// The lexer emits builtin names as plain identifiers, and the parser,
// the semantic analyzer and the error messages are all driven from this
// table, so no other file has to be touched.
//
// These names are reserved: a call to one always resolves to the builtin, so
// defining a function with the same name is rejected by the parser.

enum class BuiltinArgType {
    TENSOR,
    SCALAR
};

struct BuiltinOpSpec
{
    std::string name;
    size_t arity;              // exact number of arguments the call must have
    std::vector<BuiltinArgType> arg_types; // expected type of each argument
    bool returns_tensor;       // the call result is a tensor
};

inline const std::unordered_map<std::string, BuiltinOpSpec>& builtin_ops()
{
    static const std::unordered_map<std::string, BuiltinOpSpec> ops = {
        {"relu",          {"relu",          1, {BuiltinArgType::TENSOR}, true}},
        {"softmax",       {"softmax",       2, {BuiltinArgType::TENSOR, BuiltinArgType::SCALAR}, true}},
        {"embedding",     {"embedding",     2, {BuiltinArgType::TENSOR, BuiltinArgType::TENSOR}, true}},
        {"layernorm",     {"layernorm",     4, {BuiltinArgType::TENSOR, BuiltinArgType::TENSOR, BuiltinArgType::TENSOR, BuiltinArgType::SCALAR}, true}},
        {"mha_attention", {"mha_attention", 3, {BuiltinArgType::TENSOR, BuiltinArgType::TENSOR, BuiltinArgType::TENSOR}, true}},
    };
    return ops;
}

// Making this function for modularity.
inline const BuiltinOpSpec* find_builtin_op(const std::string& name)
{
    const auto& ops = builtin_ops();
    auto it = ops.find(name);
    return (it == ops.end()) ? nullptr : &it->second;
}

#endif