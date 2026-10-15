#ifndef TLDR_BUILTIN_OPS_H
#define TLDR_BUILTIN_OPS_H

#include <string>
#include <unordered_map>

// Single source of truth for every builtin operator of the DSL.
// To add a new operator, add one entry to builtin_ops() below.
// The lexer emits builtin names as plain identifiers, and the parser,
// the semantic analyzer and the error messages are all driven from this
// table, so no other file has to be touched.
struct BuiltinOpSpec
{
    std::string name;
    size_t arity;              // exact number of arguments the call must have
    bool requires_tensor_args; // every argument must be a non-empty tensor
    bool returns_tensor;       // the call result is a tensor
};

inline const std::unordered_map<std::string, BuiltinOpSpec>& builtin_ops()
{
    static const std::unordered_map<std::string, BuiltinOpSpec> ops = {
        {"relu",          {"relu",          1, true, true}},
        {"softmax",       {"softmax",       2, true, true}},
        {"embedding",     {"embedding",     2, true, true}},
        {"layernorm",     {"layernorm",     4, true, true}},
        {"mha_attention", {"mha_attention", 3, true, true}},
    };
    return ops;
}

//making this function for modularity. Will be used for 
inline const BuiltinOpSpec* find_builtin_op(const std::string& name)
{
    const auto& ops = builtin_ops();
    auto it = ops.find(name);
    return (it == ops.end()) ? nullptr : &it->second;
}

#endif
