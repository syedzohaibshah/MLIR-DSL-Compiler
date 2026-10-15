// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
// Datatype support for the emulated NPU: element sizes, names, fp16/bf16
// conversions, and the unified load/store-as-double accessors every compute
// kernel uses. Funnelling all element access through these two functions is
// what makes the engine dtype-agnostic and turns mixed precision / datatype
// conversion into ordinary widen-then-narrow arithmetic.
//===----------------------------------------------------------------------===//

#include <math.h>
#include <string.h>

#include "emunpu/arch.h"

const char* emunpu_status_name(emunpu_status_t status) {
  switch (status) {
    case EMUNPU_OK:
      return "OK";
    case EMUNPU_ERROR_INVALID_ARGUMENT:
      return "INVALID_ARGUMENT";
    case EMUNPU_ERROR_OUT_OF_MEMORY:
      return "OUT_OF_MEMORY";
    case EMUNPU_ERROR_OUT_OF_RANGE:
      return "OUT_OF_RANGE";
    case EMUNPU_ERROR_UNIMPLEMENTED:
      return "UNIMPLEMENTED";
    case EMUNPU_ERROR_FAILED_PRECONDITION:
      return "FAILED_PRECONDITION";
    case EMUNPU_ERROR_NOT_FOUND:
      return "NOT_FOUND";
    default:
      return "UNKNOWN";
  }
}

size_t emunpu_dtype_size(emunpu_dtype_t dtype) {
  switch (dtype) {
    case EMUNPU_DT_FP32:
    case EMUNPU_DT_INT32:
      return 4;
    case EMUNPU_DT_FP16:
    case EMUNPU_DT_BF16:
    case EMUNPU_DT_INT16:
      return 2;
    case EMUNPU_DT_INT8:
    case EMUNPU_DT_UINT8:
      return 1;
    default:
      return 0;
  }
}

const char* emunpu_dtype_name(emunpu_dtype_t dtype) {
  switch (dtype) {
    case EMUNPU_DT_FP32:
      return "fp32";
    case EMUNPU_DT_FP16:
      return "fp16";
    case EMUNPU_DT_BF16:
      return "bf16";
    case EMUNPU_DT_INT32:
      return "i32";
    case EMUNPU_DT_INT16:
      return "i16";
    case EMUNPU_DT_INT8:
      return "i8";
    case EMUNPU_DT_UINT8:
      return "u8";
    default:
      return "??";
  }
}

int emunpu_dtype_is_float(emunpu_dtype_t dtype) {
  return dtype == EMUNPU_DT_FP32 || dtype == EMUNPU_DT_FP16 ||
         dtype == EMUNPU_DT_BF16;
}

int emunpu_dtype_is_signed_int(emunpu_dtype_t dtype) {
  return dtype == EMUNPU_DT_INT32 || dtype == EMUNPU_DT_INT16 ||
         dtype == EMUNPU_DT_INT8;
}

//===----------------------------------------------------------------------===//
// Half-precision conversions
//
// IEEE-754 binary16 (fp16: 1-5-10) and bfloat16 (bf16: 1-8-7, the truncated top
// 16 bits of an fp32). Round-to-nearest-even for both. Implemented with bit
// manipulation rather than compiler _Float16 so the simulator builds anywhere.
//===----------------------------------------------------------------------===//

float emunpu_fp16_to_f32(uint16_t bits) {
  uint32_t sign = (uint32_t)(bits & 0x8000u) << 16;
  uint32_t exp = (bits >> 10) & 0x1Fu;
  uint32_t mant = bits & 0x3FFu;
  uint32_t f;
  if (exp == 0) {
    if (mant == 0) {
      f = sign;  // +/- zero
    } else {
      // Subnormal fp16: normalize into an fp32 normal.
      exp = 1;
      while ((mant & 0x400u) == 0) {
        mant <<= 1;
        exp--;
      }
      mant &= 0x3FFu;
      f = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
    }
  } else if (exp == 0x1Fu) {
    f = sign | 0x7F800000u | (mant << 13);  // inf / NaN
  } else {
    f = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
  }
  float out;
  memcpy(&out, &f, sizeof(out));
  return out;
}

uint16_t emunpu_f32_to_fp16(float value) {
  uint32_t f;
  memcpy(&f, &value, sizeof(f));
  uint32_t sign = (f >> 16) & 0x8000u;
  int32_t exp = (int32_t)((f >> 23) & 0xFFu) - 127 + 15;
  uint32_t mant = f & 0x7FFFFFu;
  if (((f >> 23) & 0xFFu) == 0xFFu) {
    // inf / NaN: preserve NaN-ness.
    return (uint16_t)(sign | 0x7C00u | (mant ? 0x200u : 0u));
  }
  if (exp >= 0x1F) {
    return (uint16_t)(sign | 0x7C00u);  // overflow -> inf
  }
  if (exp <= 0) {
    if (exp < -10) return (uint16_t)sign;  // underflow -> +/- zero
    // Subnormal: add the implicit bit and round.
    mant |= 0x800000u;
    uint32_t shift = (uint32_t)(14 - exp);
    uint32_t round_bit = 1u << (shift - 1);
    uint32_t half = mant & (round_bit - 1);
    uint32_t result = mant >> shift;
    if ((mant & round_bit) && (half != 0 || (result & 1))) result++;
    return (uint16_t)(sign | result);
  }
  // Normal: round-to-nearest-even on the 13 dropped mantissa bits.
  uint32_t round_bit = 0x1000u;
  uint32_t half = mant & (round_bit - 1);
  uint32_t result = (uint32_t)(exp << 10) | (mant >> 13);
  if ((mant & round_bit) && (half != 0 || (result & 1))) result++;
  return (uint16_t)(sign | result);
}

float emunpu_bf16_to_f32(uint16_t bits) {
  uint32_t f = (uint32_t)bits << 16;
  float out;
  memcpy(&out, &f, sizeof(out));
  return out;
}

uint16_t emunpu_f32_to_bf16(float value) {
  uint32_t f;
  memcpy(&f, &value, sizeof(f));
  if (((f >> 23) & 0xFFu) == 0xFFu) {
    // inf / NaN: keep the high mantissa bit so NaNs stay NaNs.
    return (uint16_t)((f >> 16) | (((f & 0x7FFFFFu) != 0) ? 0x40u : 0u));
  }
  // Round-to-nearest-even on the low 16 bits.
  uint32_t rounding = 0x7FFFu + ((f >> 16) & 1u);
  f += rounding;
  return (uint16_t)(f >> 16);
}

//===----------------------------------------------------------------------===//
// Unified element load/store
//===----------------------------------------------------------------------===//

double emunpu_dtype_load(const void* base, size_t index, emunpu_dtype_t dtype) {
  switch (dtype) {
    case EMUNPU_DT_FP32:
      return (double)((const float*)base)[index];
    case EMUNPU_DT_FP16:
      return (double)emunpu_fp16_to_f32(((const uint16_t*)base)[index]);
    case EMUNPU_DT_BF16:
      return (double)emunpu_bf16_to_f32(((const uint16_t*)base)[index]);
    case EMUNPU_DT_INT32:
      return (double)((const int32_t*)base)[index];
    case EMUNPU_DT_INT16:
      return (double)((const int16_t*)base)[index];
    case EMUNPU_DT_INT8:
      return (double)((const int8_t*)base)[index];
    case EMUNPU_DT_UINT8:
      return (double)((const uint8_t*)base)[index];
    default:
      return 0.0;
  }
}

// Clamps |v| to the representable range of an integer type before narrowing
// (saturating conversion, what real NPU datapaths do on downcast).
static double emunpu_clamp(double v, double lo, double hi) {
  if (v < lo) return lo;
  if (v > hi) return hi;
  return v;
}

void emunpu_dtype_store(void* base, size_t index, emunpu_dtype_t dtype,
                        double value) {
  switch (dtype) {
    case EMUNPU_DT_FP32:
      ((float*)base)[index] = (float)value;
      break;
    case EMUNPU_DT_FP16:
      ((uint16_t*)base)[index] = emunpu_f32_to_fp16((float)value);
      break;
    case EMUNPU_DT_BF16:
      ((uint16_t*)base)[index] = emunpu_f32_to_bf16((float)value);
      break;
    case EMUNPU_DT_INT32:
      ((int32_t*)base)[index] =
          (int32_t)emunpu_clamp(round(value), -2147483648.0, 2147483647.0);
      break;
    case EMUNPU_DT_INT16:
      ((int16_t*)base)[index] =
          (int16_t)emunpu_clamp(round(value), -32768.0, 32767.0);
      break;
    case EMUNPU_DT_INT8:
      ((int8_t*)base)[index] =
          (int8_t)emunpu_clamp(round(value), -128.0, 127.0);
      break;
    case EMUNPU_DT_UINT8:
      ((uint8_t*)base)[index] = (uint8_t)emunpu_clamp(round(value), 0.0, 255.0);
      break;
    default:
      break;
  }
}
