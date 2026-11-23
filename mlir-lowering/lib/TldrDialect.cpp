#include "tldr/TldrDialect.h"
#include "tldr/TldrOps.h"

using namespace tldr;

#include "tldr/TldrOpsDialect.cpp.inc"

void TldrDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "tldr/TldrOps.cpp.inc"
      >();
}
