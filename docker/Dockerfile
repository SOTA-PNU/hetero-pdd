# syntax=docker/dockerfile:1.7
FROM ubuntu:22.04

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        git \
        libaio1 \
        libglib2.0-0 \
        libgl1 \
        libgomp1 \
        libibverbs1 \
        libnuma1 \
        python3 \
        python3.10 \
        python3.10-dev \
        python3.10-venv \
        unzip \
        util-linux \
    && rm -rf /var/lib/apt/lists/*

RUN python3.10 -m venv /opt/pdd-prefill \
    && python3.10 -m venv /opt/pdd-decode \
    && /opt/pdd-prefill/bin/python -m pip install --no-cache-dir --upgrade pip==26.2.1 \
    && /opt/pdd-decode/bin/python -m pip install --no-cache-dir --upgrade pip==26.2.1

COPY docker/requirements-prefill.txt docker/requirements-decode.txt /tmp/

# pdd-prefill
RUN /opt/pdd-prefill/bin/python -m pip install --no-cache-dir --no-deps \
        -r /tmp/requirements-prefill.txt

# pdd-decode 
RUN --mount=type=bind,from=wheels,source=rebel_compiler-0.11.1.post2.dev2+g2995098f.prod-cp310-cp310-manylinux_2_34_x86_64.whl,target=/tmp/rebel_compiler-0.11.1.post2.dev2+g2995098f.prod-cp310-cp310-manylinux_2_34_x86_64.whl \
    echo '1ffab5d17fc7ed7fbbdc89503b30d8702bfa2f648b77c9fafaf8bd0a11048494  /tmp/rebel_compiler-0.11.1.post2.dev2+g2995098f.prod-cp310-cp310-manylinux_2_34_x86_64.whl' | sha256sum --check \
    && /opt/pdd-decode/bin/python -m pip install --no-cache-dir --no-deps \
        /tmp/rebel_compiler-0.11.1.post2.dev2+g2995098f.prod-cp310-cp310-manylinux_2_34_x86_64.whl

RUN /opt/pdd-decode/bin/python -m pip install --no-cache-dir --no-deps \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        --extra-index-url https://wheels.vllm.ai/0.22.0/cpu \
        -r /tmp/requirements-decode.txt

COPY . /opt/pdd-build-source
RUN /opt/pdd-decode/bin/python -m pip install --no-cache-dir --no-deps \
        /opt/pdd-build-source \
    && rm -rf /opt/pdd-build-source

COPY . /workspace
WORKDIR /workspace
RUN mkdir -p /tmp/pdd-home && chmod 1777 /tmp/pdd-home

ENV HOME=/tmp/pdd-home \
    PYTHONNOUSERSITE=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONHASHSEED=0 \
    CUDA_PY=/opt/pdd-prefill/bin/python \
    RBLN_PY=/opt/pdd-decode/bin/python \
    PYTHONPATH=/workspace \
    PDD_TOOLS_CACHE=/opt/pdd-tools \
    GIT_CONFIG_COUNT=1 \
    GIT_CONFIG_KEY_0=safe.directory \
    GIT_CONFIG_VALUE_0=/workspace

RUN bash -c 'source ./profile_tools_env.sh --install'

RUN /opt/pdd-prefill/bin/python -c 'import importlib.metadata as m, torch; assert m.version("vllm") == "0.22.0"; assert torch.version.cuda == "13.0"' \
    && /opt/pdd-decode/bin/python -c 'import importlib.metadata as m, torch; assert m.version("vllm") == "0.22.0+cpu"; assert m.version("rebel-compiler") == "0.11.1.post2.dev2+g2995098f.prod"; assert torch.version.cuda is None'

# Build-time RUN steps execute as root with HOME=/tmp/pdd-home; reset it so any --user can write.
RUN rm -rf /tmp/pdd-home && install -d -m 1777 /tmp/pdd-home

CMD ["bash"]
