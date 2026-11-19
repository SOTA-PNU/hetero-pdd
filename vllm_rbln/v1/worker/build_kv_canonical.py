# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Rebellions Inc. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Build the optional CA22 KV import helper once in the target RBLN environment."""

import argparse
import array
import ctypes
import hashlib
import importlib.util
import os
import subprocess
import tempfile
from pathlib import Path


def default_library() -> Path:
    return Path.home() / ".cache" / "hetero-pdd" / "ca22-canonical-v1.so"


# Mapping evaluated across all IEEE FP16 encodings on the pinned RBLN DSO.
# The SDK batch-produced physical table has SHA256 cf65c612903e610a9bb3d897
# c819dd268bb51a0aa4d0f750f73913bbd6c63617 with either set of logical inputs.
_CANONICAL_DIGEST = "2ae910694eff2a74613a2064a7078c57d118d0d1dc4e80a0a2fcf73dcb9d4f86"


def verified_mapping(lib: ctypes.CDLL) -> bool:
    inputs = array.array("H", range(65536))
    output = array.array("H", [0]) * len(inputs)
    lib.ca22_canonical_v1_prepare.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
    ]
    lib.ca22_canonical_v1_prepare.restype = None
    lib.ca22_canonical_v1_prepare(
        inputs.buffer_info()[0], output.buffer_info()[0], len(inputs)
    )
    return hashlib.sha256(output).hexdigest() == _CANONICAL_DIGEST


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=default_library())
    args = parser.parse_args()
    tvm = importlib.util.find_spec("tvm")
    rebel = importlib.util.find_spec("rebel")
    if not tvm or not tvm.origin or not rebel or not rebel.origin:
        parser.error("the RBLN SDK and TVM packages are required")
    sdk_lib = Path(tvm.origin).parent
    include = Path(rebel.origin).parent / "include"
    output = args.output.expanduser().absolute()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as build:
        temporary = Path(build) / output.name
        subprocess.run(
            [
                os.environ.get("CXX", "c++"),
                "-std=c++17",
                "-O3",
                "-fPIC",
                "-shared",
                str(Path(__file__).with_name("native") / "kv_canonical.cc"),
                "-I" + str(include),
                "-L" + str(sdk_lib),
                "-lrbln",
                "-ldl",
                "-Wl,-rpath," + str(sdk_lib),
                "-o",
                str(temporary),
            ],
            check=True,
        )
        lib = ctypes.CDLL(str(temporary))
        lib.ca22_canonical_v1_supported.restype = ctypes.c_bool
        if not lib.ca22_canonical_v1_supported():
            parser.error("the loaded RBLN runtime build has no validated FP16 mapping")
        if not verified_mapping(lib):
            parser.error("exhaustive IEEE FP16 preparation check failed")
        os.replace(temporary, output)
    print(output)


if __name__ == "__main__":
    main()
