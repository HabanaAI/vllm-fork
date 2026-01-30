# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.logger import init_logger
from vllm.reasoning import ReasoningParserManager
from vllm.reasoning.holo2_reasoning_parser import Holo2ReasoningParser

logger = init_logger(__name__)


@ReasoningParserManager.register_module("glm45")
class Glm4MoeModelReasoningParser(Holo2ReasoningParser):
    """
    Reasoning parser for the Glm4MoeModel model,which inherits from
    `Holo2ReasoningParser`.
    """

    pass