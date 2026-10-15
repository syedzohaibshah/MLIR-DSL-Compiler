// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
// Symbol visibility control for the emunpu shared library.
//
// EMUNPU_API annotates every public function. The library is built with hidden
// default visibility (-fvisibility=hidden), so only functions carrying this
// attribute are exported from libemunpu.so / emunpu.dll — internal helpers
// (all `static`) and any incidental symbols stay private. This keeps the ABI
// surface exactly the documented device/runtime API.
//
// Consumers need no special defines: EMUNPU_API resolves to the correct
// import/visibility form automatically. When linking against the *static*
// library, define EMUNPU_STATIC to strip the attribute entirely.
//===----------------------------------------------------------------------===//

#ifndef EMUNPU_EXPORT_H_
#define EMUNPU_EXPORT_H_

#if defined(EMUNPU_STATIC)
// Static library / amalgamated build: no export/import decoration.
#define EMUNPU_API

#elif defined(_WIN32) || defined(__CYGWIN__)
// Windows DLL: export while building the library, import when consuming it.
#if defined(EMUNPU_BUILDING_SHARED)
#define EMUNPU_API __declspec(dllexport)
#else
#define EMUNPU_API __declspec(dllimport)
#endif  // EMUNPU_BUILDING_SHARED

#elif defined(__GNUC__) || defined(__clang__)
// ELF / Mach-O: mark exported symbols as default-visible against the library's
// hidden default. Harmless (a no-op) for consumers.
#define EMUNPU_API __attribute__((visibility("default")))

#else
#define EMUNPU_API
#endif

#endif  // EMUNPU_EXPORT_H_
