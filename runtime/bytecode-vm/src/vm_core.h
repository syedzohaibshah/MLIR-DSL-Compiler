#pragma once
#include <vector>
#include <string>
#include "bytecode.h"
#include "io_handle.h"
#include "hal/device.h"
#include <unordered_map>
#include <map>
#include <set>
#include <climits>
#include "runtime_limits.h"


// One entry per device buffer the VM has allocated. A freed slot is marked
// dead, never erased, so the vector index keeps matching the handle.

struct BufferSlot {
    int  size;   // bytes ALLOC asked for
    bool live;
    int  alloc_line;
};

// What execute() does with a buffer still live at the end of a run.
// STRICT: an error. WARN_AND_FREE: a warning, then the VM frees it.
enum class LeakPolicy { STRICT, WARN_AND_FREE };

class VM{
public:
    Error setInput(int slot, void* data, int size);
    Error getOutput(int slot, void* out_ptr, int size);
    Error execute(const std::vector<opcode>& opcodes);

    // Copies the program's declared slots in header into the VM. Call before setInput.
    Error bindSignature(const Bytecode& bytecode);

    void setLeakPolicy(LeakPolicy policy);

    VM(Device* device, RuntimeLimits limits = RuntimeLimits());
    const RuntimeLimits runtime_limits;

private:
  
    IOHandler io;  // Manages input/output slots and their data
    Device *device;

    // These are the slots that the program declares in its header.
    std::map<int, SlotDecl> bound_inputs;
    std::map<int, SlotDecl> bound_outputs;

    //  slots that a successful copy used in the current run.
    std::set<int> used_inputs;
    std::set<int> used_outputs;
   
    bool signature_bound = false;
    std::unordered_map<int,int> registers; //registers are stored in a map, key is register id and value is register value

    // Buffer table: the index IS the device handle (append-only contract,
    // see device.h). Never shrinks, so the live count is counted, not derived.
    std::vector<BufferSlot> buffers;
    int live_buffer_count = 0; 	//How many are live buffers right now. Grows and shrinks.
    LeakPolicy leak_policy = LeakPolicy::STRICT;

    // Runs at the end of execute(). Handles every buffer still live (see LeakPolicy).
    Error handleLiveBuffers();

    // Device-independent buffer checks (rule D7).
    Error allocateBuffer(int size, int line_number);
    Error freeBuffer(int handle, int line_number);
    // Confirms the handle names a live buffer, and gives back its size.
    Error checkBuffer(int handle, int& out_size, int line_number) const;
    Error checkSize(int size, int line) const;

    // Checks the bound signature: `slot` is declared in `decls`, and `size` equals
    // its declared bytes.
    Error checkDeclared(const std::map<int, SlotDecl>& decls, const std::string& kind,
                        int slot, int size, int line) const;

    // after a run, every declared slot must be in its used set.

    Error checkDeclaredSlotsUsed() const;

    Error copyHostToDevice(int desti, int src_slot, int size, int line_number);
    Error copyDeviceToHost(int dest_slot, int src, int size, int line_number);
    Error dispatch(const std::string& name, const std::vector<Arg>& args,
               int line_number, int index, int opcode_size, int& next_index);


    //registers
    Error writeRegister(int id,int value,int line_number);
    Error readRegister(int id,int &value,int line_number);
    Error decrementRegister(int reg_id,int line_number);
    Error incrementRegister(int reg_id,int line_number);
      // if arg.is_register, read the register; else use the literal.
    Error resolveValue(const Arg& arg, int& out, int line);

};
