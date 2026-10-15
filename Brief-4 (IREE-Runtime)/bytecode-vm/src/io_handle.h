#pragma once
#include <vector>
#include <cstdint>

struct Input{
    void* data;
    int size;
};

class IOHandler{
public:
    // called  before running the bytecode
    void setInput(int slot, void* data, int size);

    // called after the bytecode finishes
    bool getOutput(int slot, void* out_ptr, int size);

    // validates an input slot before COPY_H2D reads from it.
    bool checkInputSlot(int slot, int expected_size, int line_number);

    // Only call after checkInputSlot() has already confirmed the slot is valid.
    void* inputData(int slot);

    // VM-facing: grows/resizes the output slot to fit `size` bytes and returns
    // a raw pointer for COPY_D2H to cudaMemcpy the result into directly.
    uint8_t* prepareOutputSlot(int slot, int size);

private:
    std::vector<Input> input_slots;
    std::vector<std::vector<uint8_t>> output_slots;
};
