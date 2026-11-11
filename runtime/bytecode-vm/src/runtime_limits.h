#pragma once

struct RuntimeLimits{
     int MAX_INPUT_SLOTS = 256;  // Maximum number of input slots allowed
     int MAX_OUTPUT_SLOTS = 256; // Maximum number of output slots allowed
     int MAX_REGISTERS = 1024;   // Maximum number of registers allowed

     int  STEP_LIMIT = 100000000 ; // (1e8) Maximum number of steps allowed during execution to prevent infinite loops
     int MAX_LIVE_BUFFERS = 4096; // if device do not provide a limit, use this default
                                  // Maximum number of live buffers allowed 
};