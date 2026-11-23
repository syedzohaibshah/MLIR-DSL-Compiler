#ifndef TLDR_DIALECT
#define TLDR_DIALECT

include "mlir/IR/OpBase.td"

def Tldr_Dialect : Dialect {
  let name = "tldr";
  let summary = "Ops mirroring the emunpu runtime's fixed LAUNCH kernel table";
  let description = [{
    The runtime (runtime/bytecode-vm) can only execute a fixed, small set of
    kernels (see runtime/doc/bytecode-contract.md's LAUNCH table) — there is
    no general-purpose compute in the bytecode VM, only ALLOC/COPY/LAUNCH of
    one of these kernels. This dialect has one op per kernel the runtime
    actually implements. It deliberately does not model the DSL's loops or
    subscripts generically: since the runtime cannot execute an arbitrary
    loop nest anyway, there is nothing to gain from a more general IR here.
  }];
  let cppNamespace = "::tldr";
}

class Tldr_Op<string mnemonic, list<Trait> traits = []> :
    Op<Tldr_Dialect, mnemonic, traits>;

#endif // TLDR_DIALECT
