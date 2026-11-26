// Emits bytecode-contract.md text from tldr-dialect MLIR. This is the other
// half of the middle: tldr-import goes JSON AST -> MLIR, this tool goes
// MLIR -> the exact text format runtime/bytecode-vm reads.
//
// Deliberately narrow, same spirit as tldr-import: only tldr.relu and
// tldr.softmax are wired up as LAUNCH-able ops right now. Anything else in
// the function body is a named error, not a guess.

#include "tldr/TldrDialect.h"
#include "tldr/TldrOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/raw_ostream.h"

#include <string>
#include <vector>

using namespace mlir;

namespace {

[[noreturn]] void die(const llvm::Twine &message) {
  llvm::errs() << "tldr-export: error: " << message << "\n";
  std::exit(1);
}

int64_t elementSizeBytes(Type elemType) {
  if (elemType.isF32() || elemType.isInteger(32))
    return 4;
  if (elemType.isF64())
    return 8;
  die("unsupported element type for sizing");
}

int64_t byteSizeOf(Type type) {
  auto tensorType = llvm::dyn_cast<RankedTensorType>(type);
  if (!tensorType || !tensorType.hasStaticShape())
    die("tldr-export only supports statically-shaped tensors right now");
  return tensorType.getNumElements() * elementSizeBytes(tensorType.getElementType());
}

struct HeaderSlot {
  int slot;
  int64_t bytes;
  std::string name;
};

// Owns the running program: buffer ids in ALLOC order, the SSA-value -> live
// buffer map, and the plain instruction text (ALLOC/COPY/LAUNCH/FREE/EOF),
// which the runtime numbers by file position -- we never compute indices
// ourselves, there is no control flow here to need them for.
struct Emitter {
  int nextBufferId = 0;
  llvm::DenseMap<Value, int> bufferOf;
  std::vector<std::string> instructions;

  int emitAlloc(int64_t bytes) {
    int id = nextBufferId++;
    instructions.push_back("ALLOC " + std::to_string(bytes));
    return id;
  }

  void emitCopyH2D(int bufferId, int slot, int64_t bytes) {
    instructions.push_back("COPY_H2D " + std::to_string(bufferId) + " " +
                            std::to_string(slot) + " " + std::to_string(bytes));
  }

  void emitCopyD2H(int slot, int bufferId, int64_t bytes) {
    instructions.push_back("COPY_D2H " + std::to_string(slot) + " " +
                            std::to_string(bufferId) + " " + std::to_string(bytes));
  }

  void emitLaunch(const std::string &argsLine) {
    instructions.push_back("LAUNCH " + argsLine);
  }

  void emitFree(int bufferId) {
    instructions.push_back("FREE " + std::to_string(bufferId));
  }
};

int lookupBuffer(const Emitter &emitter, Value v) {
  auto it = emitter.bufferOf.find(v);
  if (it == emitter.bufferOf.end())
    die("internal error: value used before its buffer was ever assigned");
  return it->second;
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 2) {
    llvm::errs() << "usage: tldr-export <module.mlir>\n";
    return 1;
  }

  MLIRContext context;
  context.getOrLoadDialect<tldr::TldrDialect>();
  context.getOrLoadDialect<func::FuncDialect>();
  context.getOrLoadDialect<linalg::LinalgDialect>();
  context.getOrLoadDialect<tensor::TensorDialect>();

  OwningOpRef<ModuleOp> module =
      parseSourceFile<ModuleOp>(argv[1], &context);
  if (!module)
    die(std::string("could not parse '") + argv[1] + "'");

  func::FuncOp funcOp;
  int funcCount = 0;
  for (auto f : module->getOps<func::FuncOp>()) {
    funcOp = f;
    ++funcCount;
  }
  if (funcCount != 1)
    die("tldr-export expects exactly one func.func in the module, found " +
        std::to_string(funcCount));

  Emitter emitter;
  std::vector<HeaderSlot> headerInputs;
  std::vector<HeaderSlot> headerOutputs;

  int nextInputSlot = 0;
  Block &entry = funcOp.getBody().front();
  for (unsigned i = 0; i < entry.getNumArguments(); ++i) {
    auto nameAttr = funcOp.getArgAttrOfType<StringAttr>(i, "tldr.name");
    if (!nameAttr)
      die("argument #" + std::to_string(i) +
          " has no tldr.name attribute -- was this produced by tldr-import?");
    bool isOutput = static_cast<bool>(funcOp.getArgAttr(i, "tldr.output"));
    if (isOutput)
      continue; // output buffers come from whatever op actually produces
                // them, not from an upfront ALLOC for the argument itself.

    Value arg = entry.getArgument(i);
    int64_t bytes = byteSizeOf(arg.getType());
    int slot = nextInputSlot++;
    int bufferId = emitter.emitAlloc(bytes);
    emitter.emitCopyH2D(bufferId, slot, bytes);
    emitter.bufferOf[arg] = bufferId;
    headerInputs.push_back({slot, bytes, nameAttr.str()});
  }

  func::ReturnOp returnOp;
  for (Operation &op : entry.without_terminator()) {
    if (auto relu = llvm::dyn_cast<tldr::ReluOp>(op)) {
      int inputBuffer = lookupBuffer(emitter, relu.getInput());
      auto inputType = llvm::cast<RankedTensorType>(relu.getInput().getType());
      int64_t count = inputType.getNumElements();
      int64_t outBytes = byteSizeOf(relu.getResult().getType());
      int outputBuffer = emitter.emitAlloc(outBytes);
      emitter.emitLaunch("2 2 " + std::to_string(inputBuffer) + " " +
                          std::to_string(outputBuffer) + " 1 " +
                          std::to_string(count));
      emitter.bufferOf[relu.getResult()] = outputBuffer;
      continue;
    }
    // LAUNCH op_id 5 (SOFTMAX). Scalars are rows, width, log_softmax, in
    // that order -- see hal/emunpu_device.cpp's do_softmax. rows/width come
    // from the input's shape, not a scalar: SoftmaxOp::verify() already
    // guarantees rank 2 by the time this tool ever sees it.
    if (auto softmax = llvm::dyn_cast<tldr::SoftmaxOp>(op)) {
      int inputBuffer = lookupBuffer(emitter, softmax.getInput());
      auto inputType = llvm::cast<RankedTensorType>(softmax.getInput().getType());
      int64_t rows = inputType.getDimSize(0);
      int64_t width = inputType.getDimSize(1);
      int64_t outBytes = byteSizeOf(softmax.getResult().getType());
      int outputBuffer = emitter.emitAlloc(outBytes);
      emitter.emitLaunch("5 2 " + std::to_string(inputBuffer) + " " +
                          std::to_string(outputBuffer) + " 3 " +
                          std::to_string(rows) + " " + std::to_string(width) +
                          " " + std::to_string(softmax.getLogSoftmax() ? 1 : 0));
      emitter.bufferOf[softmax.getResult()] = outputBuffer;
      continue;
    }
    die("tldr-export does not lower op '" + op.getName().getStringRef().str() +
        "' yet (only tldr.relu/tldr.softmax are wired up so far)");
  }
  returnOp = llvm::cast<func::ReturnOp>(entry.getTerminator());

  auto outputNamesAttr =
      funcOp->getAttrOfType<ArrayAttr>("tldr.output_names");
  if (!outputNamesAttr)
    die("function has no tldr.output_names attribute -- was this produced "
        "by tldr-import?");
  if (outputNamesAttr.size() != returnOp.getNumOperands())
    die("tldr.output_names has a different length than the return statement");

  for (unsigned i = 0; i < returnOp.getNumOperands(); ++i) {
    Value retVal = returnOp.getOperand(i);
    std::string name = llvm::cast<StringAttr>(outputNamesAttr[i]).str();
    int64_t bytes = byteSizeOf(retVal.getType());
    int bufferId = lookupBuffer(emitter, retVal);
    int slot = static_cast<int>(i);
    emitter.emitCopyD2H(slot, bufferId, bytes);
    headerOutputs.push_back({slot, bytes, name});
  }

  for (int id = 0; id < emitter.nextBufferId; ++id)
    emitter.emitFree(id);

  llvm::outs() << ".version 1\n";
  for (const HeaderSlot &in : headerInputs)
    llvm::outs() << ".input  " << in.slot << " " << in.bytes << " " << in.name
                 << "\n";
  for (const HeaderSlot &out : headerOutputs)
    llvm::outs() << ".output " << out.slot << " " << out.bytes << " "
                 << out.name << "\n";
  llvm::outs() << "\n";

  int idx = 0;
  for (const std::string &line : emitter.instructions)
    llvm::outs() << line << "    // " << idx++ << "\n";
  llvm::outs() << "EOF    // " << idx << "\n";

  return 0;
}
