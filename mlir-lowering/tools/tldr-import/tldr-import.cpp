// Imports the DSL frontend's JSON AST (Brief-1 DSL Frontend --json) into the
// tldr dialect. Deliberately narrow: it only understands the exact shapes
// Phase 2's minimal slice targets (a function body made of plain Assign
// statements whose right-hand side is a single BuiltinCall). Anything else
// is a clean, named error -- never a silent misinterpretation.

#include "tldr/TldrDialect.h"
#include "tldr/TldrOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/raw_ostream.h"

#include "json.hpp"

#include <fstream>
#include <set>
#include <string>
#include <vector>

using json = nlohmann::json;
using namespace mlir;

namespace {

[[noreturn]] void die(const llvm::Twine &message) {
  llvm::errs() << "tldr-import: error: " << message << "\n";
  std::exit(1);
}

Type dtypeToElementType(const std::string &dtype, MLIRContext *ctx) {
  if (dtype == "f32")
    return Float32Type::get(ctx);
  if (dtype == "f64")
    return Float64Type::get(ctx);
  if (dtype == "i32")
    return IntegerType::get(ctx, 32);
  die("unsupported dtype '" + dtype + "'");
}

// A DSL parameter is either a tensor (has "shape") or a scalar (does not).
Type paramType(const json &param, MLIRContext *ctx) {
  Type elemType = dtypeToElementType(param.at("dtype").get<std::string>(), ctx);
  if (!param.contains("shape"))
    return elemType;

  llvm::SmallVector<int64_t> shape;
  for (const json &dim : param.at("shape")) {
    if (!dim.is_number_integer())
      die("symbolic shape dimensions are not supported by the importer yet "
          "(param '" +
          param.at("name").get<std::string>() + "')");
    shape.push_back(dim.get<int64_t>());
  }
  return RankedTensorType::get(shape, elemType);
}

Value lookupVariable(const json &node, const llvm::StringMap<Value> &symtab) {
  if (node.at("kind").get<std::string>() != "Variable")
    die("the importer only supports a plain variable as a builtin-call "
        "argument right now, got AST kind '" +
        node.at("kind").get<std::string>() + "'");
  std::string name = node.at("name").get<std::string>();
  auto it = symtab.find(name);
  if (it == symtab.end())
    die("variable '" + name + "' is used before it is assigned");
  return it->second;
}

Value lowerBuiltinCall(OpBuilder &builder, Location loc, const json &call,
                        const llvm::StringMap<Value> &symtab) {
  std::string op = call.at("op").get<std::string>();
  const json &args = call.at("args");

  if (op == "relu") {
    if (args.size() != 1)
      die("relu expects exactly 1 argument, got " +
          std::to_string(args.size()));
    Value input = lookupVariable(args[0], symtab);
    return builder.create<tldr::ReluOp>(loc, input.getType(), input);
  }

  die("the importer does not lower builtin op '" + op +
      "' yet (only relu is wired up so far)");
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 2) {
    llvm::errs() << "usage: tldr-import <ast.json>\n";
    return 1;
  }

  std::ifstream in(argv[1]);
  if (!in)
    die(std::string("could not open '") + argv[1] + "'");

  json root;
  try {
    in >> root;
  } catch (const json::parse_error &e) {
    die(std::string("failed to parse JSON: ") + e.what());
  }

  MLIRContext context;
  context.getOrLoadDialect<tldr::TldrDialect>();
  context.getOrLoadDialect<func::FuncDialect>();

  OpBuilder builder(&context);
  Location loc = builder.getUnknownLoc();
  auto module = ModuleOp::create(loc);
  builder.setInsertionPointToEnd(module.getBody());

  std::string funcname = root.at("funcname").get<std::string>();
  const json &params = root.at("params");
  const json &returnsJson = root.at("returns");

  std::vector<Type> argTypes;
  std::vector<std::string> argNames;
  for (const json &p : params) {
    argTypes.push_back(paramType(p, &context));
    argNames.push_back(p.at("name").get<std::string>());
  }

  std::vector<std::string> outputOrder;
  for (const json &r : returnsJson)
    outputOrder.push_back(r.get<std::string>());
  std::set<std::string> outputSet(outputOrder.begin(), outputOrder.end());

  std::vector<Type> resultTypes;
  for (const std::string &name : outputOrder) {
    auto it = std::find(argNames.begin(), argNames.end(), name);
    if (it == argNames.end())
      die("return name '" + name +
          "' is not a parameter -- the importer only supports a return name "
          "that is also an input/output parameter right now");
    resultTypes.push_back(argTypes[std::distance(argNames.begin(), it)]);
  }

  auto funcType = builder.getFunctionType(argTypes, resultTypes);
  auto funcOp = builder.create<func::FuncOp>(loc, funcname, funcType);
  Block *entry = funcOp.addEntryBlock();

  // MLIR's own printed form only keeps %arg0, %arg1, ... -- the DSL's real
  // parameter names ("A", "B", ...) would otherwise be lost here forever.
  // The bytecode-contract header needs those names, so they're carried as
  // explicit attributes rather than relied on implicitly.
  llvm::StringMap<Value> symtab;
  for (size_t i = 0; i < argNames.size(); ++i) {
    symtab[argNames[i]] = entry->getArgument(i);
    funcOp.setArgAttr(i, "tldr.name", builder.getStringAttr(argNames[i]));
    if (outputSet.count(argNames[i]))
      funcOp.setArgAttr(i, "tldr.output", builder.getUnitAttr());
  }
  // Likewise, "which function result is named what" only exists in the
  // JSON's "returns" list right now; outputOrder is that list, in exactly
  // the order the function's result types (and so `return`'s operands)
  // were built from above, so it can be attached as-is.
  llvm::SmallVector<llvm::StringRef> outputOrderRefs(outputOrder.begin(),
                                                      outputOrder.end());
  funcOp->setAttr("tldr.output_names",
                   builder.getStrArrayAttr(outputOrderRefs));

  builder.setInsertionPointToStart(entry);

  for (const json &stmt : root.at("body")) {
    std::string kind = stmt.at("kind").get<std::string>();
    if (kind != "Assign")
      die("the importer only supports Assign statements right now, got AST "
          "kind '" +
          kind + "'");

    const json &target = stmt.at("target");
    if (target.at("kind").get<std::string>() != "Variable")
      die("the importer only supports assigning to a plain variable right "
          "now, not a subscript target");
    std::string targetName = target.at("name").get<std::string>();

    const json &value = stmt.at("value");
    if (value.at("kind").get<std::string>() != "BuiltinCall")
      die("the importer only supports a builtin-call right-hand side right "
          "now, got AST kind '" +
          value.at("kind").get<std::string>() + "'");

    symtab[targetName] = lowerBuiltinCall(builder, loc, value, symtab);
  }

  llvm::SmallVector<Value> retVals;
  for (const std::string &name : outputOrder) {
    auto it = symtab.find(name);
    if (it == symtab.end())
      die("output '" + name + "' was never assigned in the function body");
    retVals.push_back(it->second);
  }
  builder.create<func::ReturnOp>(loc, retVals);

  if (failed(verify(module))) {
    module.print(llvm::errs());
    return 1;
  }

  module.print(llvm::outs());
  llvm::outs() << "\n";
  return 0;
}
