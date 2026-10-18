#pragma once
#include <vector>
#include <cstdint>
#include "error.h"
struct Input{
    void* data;
    int size;
};
struct OutputSlot {
    std::vector<uint8_t> data;
    bool written_ok;   // true only after a clean download
};

class IOHandler{
public:
    // called  before running the bytecode
    Error setInput(int slot, void* data, int size);

    // called after the bytecode finishes
    Error getOutput(int slot, void* out_ptr, int size);

    // validates an input slot before COPY_H2D reads from it.
    Error checkInputSlot(int slot, int expected_size, int line_number);

    // Only call after checkInputSlot() has already confirmed the slot is valid.
    void* inputData(int slot);

    // VM-facing: grows/resizes the output slot to fit `size` bytes and returns
    // a raw pointer for COPY_D2H to cudaMemcpy the result into directly.
    uint8_t* prepareOutputSlot(int slot, int size);

    void markOutputSlotWritten(int slot);

private:
    std::vector<Input> input_slots;
   std::vector<OutputSlot> output_slots;
};
