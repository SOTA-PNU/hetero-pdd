// SPDX-License-Identifier: Apache-2.0
// Diagnostic oracle for this SDK. The internal converter is never a device
// write API. Production transfers still go through ordinary blocking copy_.
#include <cstddef>
#include <cstdint>

namespace rbln {
uint16_t convHalf2Dlfloat(uint16_t);
void ConvertFloat16ToDLFloat16(unsigned long, unsigned long, unsigned long);
void ConvertDLFloat16ToFloat16(unsigned long, unsigned long, unsigned long);
} // namespace rbln

extern "C" void canonical(const uint16_t *in, uint16_t *out, size_t count) {
  for (size_t i = 0; i < count; ++i) {
    const uint16_t x = in[i], exponent = x & 0x7c00;
    const bool eligible =
        exponent != 0 && exponent != 0x7c00 && (x & 0x7fff) != 0x7bff;
    out[i] = x + static_cast<uint16_t>(eligible && (x & 1));
  }
}

extern "C" void lanes(const uint16_t *in, uint16_t *out, size_t count) {
  for (size_t i = 0; i < count; ++i) {
    out[i] = rbln::convHalf2Dlfloat(in[i]);
  }
}
extern "C" void batch(const uint16_t *in, uint16_t *out, size_t count) {
  rbln::ConvertFloat16ToDLFloat16(reinterpret_cast<unsigned long>(in),
                                  reinterpret_cast<unsigned long>(out), count);
}
extern "C" void decode(const uint16_t *in, uint16_t *out, size_t count) {
  rbln::ConvertDLFloat16ToFloat16(reinterpret_cast<unsigned long>(in),
                                  reinterpret_cast<unsigned long>(out), count);
}
