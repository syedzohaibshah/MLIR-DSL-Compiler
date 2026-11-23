#ifndef TLDR_TLDROPS_H
#define TLDR_TLDROPS_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "tldr/TldrDialect.h"

#define GET_OP_CLASSES
#include "tldr/TldrOps.h.inc"

#endif // TLDR_TLDROPS_H
