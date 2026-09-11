# Upstream app-verl0.4-vllm0.8.5-mcore0.12.2-te2.2, linux/amd64.
FROM verlai/verl@sha256:0e90bc41d0dbadf47207c418afa625a6649941c7aec8aeb455f7e03827f311ea

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV PIP_CONFIG_FILE=/dev/null PIP_INDEX_URL=https://pypi.org/simple \
    HF_HUB_ENABLE_HF_TRANSFER=0 VLLM_USE_V1=0 TOKENIZERS_PARALLELISM=false

COPY requirements-grpo.txt /opt/probed-grpo/requirements-grpo.txt
# Preserve the binary ABI of the upstream torch/vLLM/FlashAttention stack.
COPY grpo/constraints-runtime.txt /opt/probed-grpo/constraints-runtime.txt
COPY grpo/requirements-overlay.lock /opt/probed-grpo/requirements-overlay.lock
# The upstream image intentionally uses cuDNN 9.8 with torch 2.6. Keep it intact.
RUN python -m pip install --no-cache-dir --no-deps \
    -r /opt/probed-grpo/requirements-overlay.lock

RUN git clone https://github.com/volcengine/verl.git /opt/verl && \
    git -C /opt/verl checkout --detach 8fdc4d3f202f41461f4de9f42a637228e342668b && \
    python -m pip install --no-deps --no-build-isolation -e /opt/verl
RUN python -c "import torch, vllm, flash_attn, verl; from verl.trainer import main_ppo; from verl.workers.reward_manager.batch import BatchRewardManager; assert torch.__version__.split('+')[0] == '2.6.0'; assert vllm.__version__ == '0.8.5.post1'" && \
    python -m pip freeze > /opt/probed-grpo/environment.freeze.txt

WORKDIR /workspace/probed-grpo
CMD ["bash"]
