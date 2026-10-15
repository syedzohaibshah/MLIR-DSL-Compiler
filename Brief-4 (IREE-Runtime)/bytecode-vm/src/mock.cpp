// Mock Device: runs any bytecode file without a GPU and
// logs alloc / free / launch, double-frees and leaks.
#include "vm_core.h"
#include "bytecode.h"
#include <iostream>

struct Mock : Device {
    int n = 0;
    std::vector<bool> freed;
    std::string name() const override { return "mock"; }

    DeviceHandle allocate(int size, int) override {
        std::cout << "alloc size=" << size << " -> " << n << "\n";
        freed.push_back(false);
        return n++;
    }
    void free(DeviceHandle h, int line) override {
        if (h < 0 || h >= n || freed[h])
            std::cout << "line " << line << ": BAD free " << h << "\n";
        else { freed[h] = true; std::cout << "free " << h << "\n"; }
    }
    bool upload(DeviceHandle, const void*, int, int) override {
        return true;
    }
    bool download(DeviceHandle h, void*, int, int) override {
        std::cout << "download " << h << "\n";
        return false;
    }
    bool dispatch(const std::vector<int>& a, int) override {
        std::cout << "launch";
        for (int x : a) std::cout << " " << x;
        std::cout << "\n";
        return true;
    }
    ~Mock() {
        for (int i = 0; i < n; i++)
            if (!freed[i]) std::cout << "LEAK buffer " << i << "\n";
    }
};

int main(int argc, char** argv) {
    Bytecode b;
    b.loadFromFile(argv[1]);
    Mock m;
    {
        VM vm(&m);
        float x[4]{};
        // slots 0..4 as in bytecode12: X, W0, B0, W1, B1
        for (int i = 0; i < 5; i++)
            vm.setInput(i, x, (i == 2 || i == 4) ? 8 : 16);
        vm.execute(b.get_instructions());
    }
}
