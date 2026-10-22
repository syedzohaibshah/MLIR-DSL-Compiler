#ifndef TLDR_CONST_EVAL_H
#define TLDR_CONST_EVAL_H

#include "ast_nodes.h"
#include "json.hpp"
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

using json = nlohmann::json;

// A compile-time scalar. Mirrors ConstantNode, which stores every literal as a
// double plus a flag telling whether it was written as an integer.
struct Scalar {
    double value = 0.0;
    bool is_int = false;
};

// Folds an expression built from literals and arithmetic into a single
// scalar. Returns false when the expression is not a compile-time constant.
bool eval_scalar(const std::shared_ptr<ASTNode>& node, Scalar& out);

// Static analysis of the initialiser of a tensor declaration.
struct TensorInitInfo {
    enum class Status {
        OK,             // the initialiser is a statically known tensor
        NO_INIT,        // the declaration carries no initialiser
        NOT_A_LIST,     // the initialiser is not a list literal
        SCALAR_INIT,    // a scalar value cannot initialise a tensor
        NOT_CONSTANT,   // the initialiser holds elements that are not constant
        RAGGED,         // the nested lists do not all have the same length
        RANK_MISMATCH,  // the literal nests deeper than the declared shape
        COUNT_MISMATCH  // the literal holds a different number of elements
    };

    Status status = Status::NOT_A_LIST;
    std::string message;                // human readable detail of a failure
    std::vector<Scalar> raw;            // the elements in row-major order
    std::vector<int64_t> literal_shape; // the shape of the literal itself
    std::vector<int64_t> declared_dims; // the declared dims, when resolved
    bool dims_resolved = false;         // true when every declared dim is known
};

// Flattens the initialiser of a tensor declaration into its elements and, when
// dim_bindings is given, checks the literal against the declared shape. Pass
// nullptr to only flatten, without validating against the declared dims.
TensorInitInfo analyze_tensor_init(
    const std::shared_ptr<TensorDeclNode>& node,
    const std::unordered_map<std::string, int64_t>* dim_bindings);

// Rebuilds a nested JSON array of dims from a flat list of scalars. The caller
// must have checked that raw holds product(dims) elements.
json rebuild_tensor(const std::vector<int64_t>& dims, const std::vector<Scalar>& raw);

#endif
