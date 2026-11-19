// SPDX-License-Identifier: Apache-2.0
#include <rebel/runtime/api/rbln_runtime_api.h>

#include <sstream>
#include <string>

namespace {
template <typename Shape>
void write_shape(std::ostream &out, const Shape &values) {
  out << '[';
  for (size_t i = 0; i < values.size(); ++i) {
    if (i)
      out << ',';
    out << values[i];
  }
  out << ']';
}
} // namespace

extern "C" const char *ca22_memory_info(uint64_t address) {
  rbln::MemoryInfo info{};
  if (rbln::rbln_get_memory_info(address, info) != RBLNRetCode_SUCCESS) {
    return nullptr;
  }
  std::ostringstream out;
  out << "{\"user_dtype\":" << static_cast<int>(info.user_dtype)
      << ",\"is_user_fp16\":" << std::boolalpha
      << (info.user_dtype == rbln::DataType::Float16)
      << ",\"physical_dtype\":" << static_cast<int>(info.physical_dtype)
      << ",\"is_physical_custom_fp16\":"
      << (info.physical_dtype == rbln::DataType::CustomFloat16)
      << ",\"user_shape\":";
  write_shape(out, info.user_shape);
  out << ",\"physical_shape\":";
  write_shape(out, info.physical_shape);
  out << '}';
  static thread_local std::string result;
  result = out.str();
  return result.c_str();
}
