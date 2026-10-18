#pragma once
#include <string>
#include <iostream>

struct Error{
    std::string msg;
    bool ok;
    int line_number;

};

// Builds a failure Error. ok=false.
inline Error makeError(const std::string& msg, int line_number){
    return Error{msg, false, line_number};
}

// Builds a success Error. ok=true. No message needed.
inline Error ok(){
    return Error{"", true, 0};
}

// Prints an Error. Call this once, at the top stop point, when ok is false.
inline void report(const Error& e){
    std::cerr<<"Error at line "<<e.line_number<<": "<<e.msg<<std::endl;
}

// Runs `expr` one time, stores its Error, and returns that Error from the
// current function if it failed. Use it at every call site that returns Error.
//
// The current function MUST return Error for this macro to compile.
#define RETURN_IF_ERROR(expr)                 \
    do {                                      \
        Error _status = (expr);               \
        if(!_status.ok) return _status;       \
    } while(0)
