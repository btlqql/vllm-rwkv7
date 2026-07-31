# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass

from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.attention.backends.mamba_attn import (
    BaseMambaAttentionMetadata,
    BaseMambaAttentionMetadataBuilder,
)


class RWKV7AttentionBackend(AttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "RWKV7_ATTN"

    @staticmethod
    def get_builder_cls() -> type["RWKV7AttentionMetadataBuilder"]:
        return RWKV7AttentionMetadataBuilder

    @classmethod
    def is_ssm(cls) -> bool:
        return True


@dataclass
class RWKV7AttentionMetadata(BaseMambaAttentionMetadata):
    pass


class RWKV7AttentionMetadataBuilder(
    BaseMambaAttentionMetadataBuilder[RWKV7AttentionMetadata]
):
    metadata_cls = RWKV7AttentionMetadata
