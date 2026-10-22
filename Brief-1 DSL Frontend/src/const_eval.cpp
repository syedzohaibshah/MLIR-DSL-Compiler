#include "const_eval.h"

#include <algorithm>
#include <cmath>

namespace {

// Guards against absurd dimension products overflowing int64.
constexpr int64_t kMaxElementCount = int64_t(1) << 40;

json scalar_to_json(const Scalar& s) {
    if (s.is_int) return static_cast<int64_t>(s.value);
    return s.value;
}

// Renders a resolved shape the way it is written in the source.
std::string shape_to_string(const std::vector<int64_t>& dims) {
    std::string s;
    for (size_t i = 0; i < dims.size(); ++i) {
        if (i) s += ", ";
        s += std::to_string(dims[i]);
    }
    return s;
}

// The number of elements a tensor of dims holds, or false when the dims are
// negative or the product is too large to be meaningful.
bool shape_element_count(const std::vector<int64_t>& dims, int64_t& out) {
    int64_t count = 1;
    for (int64_t d : dims) {
        if (d < 0) return false;
        if (d != 0 && count > kMaxElementCount / d) return false;
        count *= d;
    }
    out = count;
    return true;
}

// Flattens a list literal into its elements and records the shape the literal
// nests to. Returns the failure reason when the literal is not a tensor of
// constants.
TensorInitInfo::Status flatten_list(const std::shared_ptr<ListNode>& node,
                                    std::vector<Scalar>& raw,
                                    std::vector<int64_t>& shape,
                                    std::string& reason) {
    shape.clear();
    shape.push_back(static_cast<int64_t>(node->elements.size()));

    bool saw_list = false;
    bool saw_scalar = false;
    std::vector<int64_t> child_shape;

    for (const auto& elem : node->elements) {
        if (!elem) {
            reason = "is initialized with an empty element";
            return TensorInitInfo::Status::NOT_CONSTANT;
        }
        if (elem->kind() == ASTKind::LIST) {
            saw_list = true;
            std::vector<int64_t> child;
            const TensorInitInfo::Status child_status =
                flatten_list(std::static_pointer_cast<ListNode>(elem), raw, child,
                             reason);
            if (child_status != TensorInitInfo::Status::OK) {
                return child_status;
            }
            if (child_shape.empty()) {
                child_shape = child;
            } else if (child != child_shape) {
                reason = "is initialized with a ragged literal: the nested lists "
                         "do not all have the same length";
                return TensorInitInfo::Status::RAGGED;
            }
        } else {
            saw_scalar = true;
            Scalar s;
            if (!eval_scalar(elem, s)) {
                reason = "is initialized with an element that is not a "
                         "compile-time constant";
                return TensorInitInfo::Status::NOT_CONSTANT;
            }
            raw.push_back(s);
        }
    }

    if (saw_list && saw_scalar) {
        reason = "is initialized with a ragged literal: it mixes scalars and "
                 "nested lists";
        return TensorInitInfo::Status::RAGGED;
    }

    if (saw_list) {
        shape.insert(shape.end(), child_shape.begin(), child_shape.end());
    }
    return TensorInitInfo::Status::OK;
}

// Rebuilds one level of the nested array, consuming raw in row-major order.
json rebuild_level(const std::vector<int64_t>& dims, size_t dim,
                   const std::vector<Scalar>& raw, size_t& offset) {
    json j = json::array();
    const int64_t count = dims[dim];
    if (dim + 1 == dims.size()) {
        for (int64_t i = 0; i < count; ++i) {
            if (offset >= raw.size()) break;
            j.push_back(scalar_to_json(raw[offset++]));
        }
        return j;
    }
    for (int64_t i = 0; i < count; ++i) {
        j.push_back(rebuild_level(dims, dim + 1, raw, offset));
    }
    return j;
}

}  // namespace

bool eval_scalar(const std::shared_ptr<ASTNode>& node, Scalar& out) {
    if (!node) return false;

    switch (node->kind()) {
        case ASTKind::CONSTANT: {
            auto c = std::static_pointer_cast<ConstantNode>(node);
            out.value = c->value;
            out.is_int = c->is_integer;
            return true;
        }
        case ASTKind::UNARY_OP: {
            auto u = std::static_pointer_cast<UnaryOpNode>(node);
            if (u->op != "-") return false;
            Scalar operand;
            if (!eval_scalar(u->operand, operand)) return false;
            out.value = -operand.value;
            out.is_int = operand.is_int;
            return true;
        }
        case ASTKind::BINARY_OP: {
            auto b = std::static_pointer_cast<BinaryOpNode>(node);
            Scalar lhs, rhs;
            if (!eval_scalar(b->left, lhs) || !eval_scalar(b->right, rhs)) {
                return false;
            }
            const bool both_int = lhs.is_int && rhs.is_int;
            if (b->op == "+") {
                out.value = lhs.value + rhs.value;
            } else if (b->op == "-") {
                out.value = lhs.value - rhs.value;
            } else if (b->op == "*") {
                out.value = lhs.value * rhs.value;
            } else if (b->op == "/") {
                if (rhs.value == 0) return false;
                out.value = lhs.value / rhs.value;
                out.is_int = both_int && (out.value == std::trunc(out.value));
                return true;
            } else {
                return false;
            }
            out.is_int = both_int;
            return true;
        }
        default:
            return false;
    }
}

TensorInitInfo analyze_tensor_init(
    const std::shared_ptr<TensorDeclNode>& node,
    const std::unordered_map<std::string, int64_t>* dim_bindings) {
    TensorInitInfo info;
    if (!node || !node->init_value) {
        info.status = TensorInitInfo::Status::NO_INIT;
        return info;
    }
    if (node->shape.empty()) {
        // A declaration without a shape is a scalar, so it holds no tensor
        // data to check against dims.
        info.status = TensorInitInfo::Status::NO_INIT;
        return info;
    }

    if (node->init_value->kind() != ASTKind::LIST) {
        // A variable or a call may hold a tensor, so only a scalar constant is
        // reported as a mistake here.
        Scalar s;
        if (eval_scalar(node->init_value, s)) {
            info.status = TensorInitInfo::Status::SCALAR_INIT;
            info.message = "'" + node->name +
                           "' is declared as a tensor but is initialized with a "
                           "scalar value";
        } else {
            info.status = TensorInitInfo::Status::NOT_A_LIST;
        }
        return info;
    }

    std::string reason;
    const TensorInitInfo::Status status =
        flatten_list(std::static_pointer_cast<ListNode>(node->init_value),
                     info.raw, info.literal_shape, reason);
    if (status != TensorInitInfo::Status::OK) {
        info.status = status;
        info.message = "'" + node->name + "' " + reason;
        return info;
    }

    // Resolve the declared dims. A dim that is neither a literal nor bound in
    // the current scope leaves the element count unknown, so no check is done.
    std::vector<int64_t> dims;
    for (const auto& dim : node->shape) {
        if (std::holds_alternative<int64_t>(dim)) {
            dims.push_back(std::get<int64_t>(dim));
            continue;
        }
        const std::string& name = std::get<std::string>(dim);
        auto it = dim_bindings ? dim_bindings->find(name)
                               : std::unordered_map<std::string, int64_t>::const_iterator();
        if (!dim_bindings || it == dim_bindings->end()) {
            info.declared_dims = dims;
            return info;
        }
        dims.push_back(it->second);
    }
    info.declared_dims = dims;
    info.dims_resolved = true;

    if (info.literal_shape.size() != dims.size()) {
        info.status = TensorInitInfo::Status::RANK_MISMATCH;
        info.message = "'" + node->name + "' is declared with " +
                       std::to_string(dims.size()) + " dimension(s) but is "
                       "initialized with a literal of rank " +
                       std::to_string(info.literal_shape.size());
        return info;
    }

    int64_t expected = 0;
    if (!shape_element_count(dims, expected)) {
        // The declared shape is not usable, so there is nothing to check.
        return info;
    }

    if (static_cast<int64_t>(info.raw.size()) != expected) {
        info.status = TensorInitInfo::Status::COUNT_MISMATCH;
        info.message = "'" + node->name + "' is declared as Tensor[" +
                       node->dtype + ", (" + shape_to_string(dims) +
                       ")] (" + std::to_string(expected) +
                       " element(s)) but is initialized with " +
                       std::to_string(info.raw.size()) + " element(s)";
        return info;
    }

    info.status = TensorInitInfo::Status::OK;
    return info;
}

json rebuild_tensor(const std::vector<int64_t>& dims, const std::vector<Scalar>& raw) {
    if (dims.empty()) {
        // A tensor without dimensions holds a single value.
        if (raw.empty()) return json(nullptr);
        return scalar_to_json(raw[0]);
    }
    size_t offset = 0;
    return rebuild_level(dims, 0, raw, offset);
}
