# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# MIT_LICENSE file in the root directory of this source tree.
from __future__ import annotations

import logging
from argparse import ArgumentParser, Namespace
from typing import Any, Dict, List, Optional, Union

import torch
from fairseq2.assets import asset_store
from seamless_communication.inference.translator import Modality, Translator
from seamless_communication.models.generator.loader import load_pretssel_vocoder_model
from seamless_communication.models.generator.vocoder import PretsselVocoder
from seamless_communication.models.monotonic_decoder import (
    load_monotonic_decoder_config,
    load_monotonic_decoder_model,
)
from seamless_communication.models.unity import (
    load_unity_config,
    load_unity_model,
    load_unity_text_tokenizer,
    load_unity_unit_tokenizer,
)
from seamless_communication.models.vocoder.loader import load_vocoder_model
from seamless_communication.models.vocoder.vocoder import Vocoder
from seamless_communication.streaming.agents.common import (
    AgentStates,
    EarlyStoppingMixin,
)
from simuleval.agents import AgentPipeline, TreeAgentPipeline
from simuleval.agents.agent import GenericAgent
from simuleval.data.segments import Segment

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s -- %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)


def maybe_reset_states(states: Optional[List[Optional[AgentStates]]]) -> None:
    assert states is not None
    for s in states:
        if s is not None:
            if isinstance(s, EarlyStoppingMixin):
                s.reset_early()
            else:
                s.reset()


class UnitYPipelineMixin:
    """
    Mixin for UnitY pipeline which works with both AgentPipeline
    and TreeAgentPipeline
    """

    @classmethod
    def add_args(cls, parser: ArgumentParser) -> None:
        super().add_args(parser)  # type: ignore
        parser.add_argument("--task", type=str, help="Task type")
        parser.add_argument(
            "--unity-model-name",
            type=str,
            help="Unity model name.",
            default="seamless_streaming_unity",
        )
        parser.add_argument(
            "--monotonic-decoder-model-name",
            type=str,
            help="Monotonic decoder model name.",
            default="seamless_streaming_monotonic_decoder",
        )

        parser.add_argument(
            "--sample-rate",
            default=16000,
            type=float,
        )
        parser.add_argument(
            "--dtype",
            choices=["fp16", "fp32"],
            default="fp16",
            type=str,
            help=(
                "Choose between half-precision (fp16) and single precision (fp32) floating point formats."
                + " Prefer this over the fp16 flag."
            ),
        )

    @classmethod
    def load_model(cls, args: Namespace) -> Dict[str, Any]:
        if not torch.cuda.is_available() and "cuda" in args.device:
            raise ValueError("CUDA not available, use CPU.")

        # args.device = torch.device(args.device)
        args.device = torch.device('cpu')
        if (args.fp16 or args.dtype == "fp16") and args.device != torch.device("cpu"):
            args.dtype = torch.float16
        else:
            args.dtype = torch.float32

        input_modality, output_modality = Translator.get_modalities_from_task_str(
            args.task
        )

        if input_modality != Modality.SPEECH:
            raise ValueError("`UnitYAgentPipeline` only supports speech input.")

        unity_config = load_unity_config(args.unity_model_name)
        unity_config.use_text_decoder = False
        unity_config.use_text_encoder = False

        text_tokenizer = load_unity_text_tokenizer(args.unity_model_name)

        # Skip loading the T2U model.
        if output_modality == Modality.TEXT:
            unity_config.t2u_config = None
            unit_tokenizer = None
        else:
            unit_tokenizer = load_unity_unit_tokenizer(args.unity_model_name)

        asset_card = asset_store.retrieve_card(args.unity_model_name)
        asset_card.field("model_config").set(unity_config)

        logger.info(
            f"Loading the UnitY model: {args.unity_model_name} on device={args.device}, dtype={args.dtype}"
        )
        # print(asset_card)
        unity_model = load_unity_model(asset_card, device=args.device, dtype=args.dtype)
        unity_model.eval()

        print(type(unity_model.speech_encoder_frontend))
        # unity_model.text_decoder = text_tokenizer.create_decoder()
        # streaming_evaluate --task s2tt --data-file ./cvssc_ja/test.tsv --audio-root-dir ./cvssc_ja/test --output ./test --tgt-lang eng --dtype fp32 --device cpu
        """
        input_names = ['input_seqs']
        output_names = ['output_seqs']
        dynamic_axes_1 = {
            'input_seqs' : {1: 'seq_len'},
            'output_seqs' : {1: 'seq_len'},
        }
        x = (torch.randn((1, 50, 80),requires_grad=False))
        # torch.save(monotonic_decoder_model.text_decoder_frontend.state_dict(), 'seamless_streaming_monotonic_decoder_text_decoder_frontend.pt')
        torch.onnx.export(unity_model.speech_encoder_frontend.cpu(), x, 'seamless_streaming_unity_speech_encoder_frontend.onnx', input_names=input_names, output_names=output_names, verbose='True', opset_version=12, dynamic_axes=dynamic_axes_1)
        """
        """
        input_names = ['input_seqs']
        output_names = ['output_seqs']
        x = (torch.randn((1, 640, 80),requires_grad=False))
        # torch.save(monotonic_decoder_model.text_decoder_frontend.state_dict(), 'seamless_streaming_monotonic_decoder_text_decoder_frontend.pt')
        torch.onnx.export(unity_model.speech_encoder_frontend.cpu(), x, 'seamless_streaming_unity_speech_encoder_frontend.onnx', input_names=input_names, output_names=output_names, verbose='True', opset_version=12)
        """

        print(type(unity_model.speech_encoder))
        input_names = ['input_seqs', 'seq_len']
        output_names = ['output_seqs']
        x = (torch.randn(1,80, 1024,requires_grad=False), torch.tensor([27], dtype=torch.int64))
        # torch.save(monotonic_decoder_model.text_decoder_frontend.state_dict(), 'seamless_streaming_monotonic_decoder_text_decoder_frontend.pt')
        torch.onnx.export(unity_model.speech_encoder.cpu(), x, 'seamless_streaming_unity_speech_encoder.onnx', input_names=input_names, output_names=output_names, verbose='True', opset_version=13)

        

        monotonic_decoder_config = load_monotonic_decoder_config(
            args.monotonic_decoder_model_name
        )
        logger.info(
            f"Loading the Monotonic Decoder model: {args.monotonic_decoder_model_name} on device={args.device}, dtype={args.dtype}"
        )
        monotonic_decoder_model = load_monotonic_decoder_model(
            args.monotonic_decoder_model_name, device=args.device, dtype=args.dtype
        )
        monotonic_decoder_model.eval()
        """
        print(type(monotonic_decoder_model))
        # text_decoder_frontend
        input_names = ['seqs', 'padding_mask_params', 'state_bag']
        output_names = ['embeds']
        dynamic_axes_1 = {
            'seqs' : {1: 'seq_len'},
            'embeds' : {1: 'seq_len'}
        }
        x = (torch.tensor([[3, 256022]],requires_grad=False).int(), None, [4096, 0, {}])
        # torch.save(monotonic_decoder_model.text_decoder_frontend.state_dict(), 'seamless_streaming_monotonic_decoder_text_decoder_frontend.pt')
        torch.onnx.export(monotonic_decoder_model.text_decoder_frontend.cpu(), x, 'seamless_streaming_monotonic_decoder_text_decoder_frontend.onnx', input_names=input_names, output_names=output_names, verbose='True', opset_version=12, dynamic_axes=dynamic_axes_1)
        """
        # text_decoder
        """
        input_names = ['input_seqs', 'encoder_output', 'state_bag']
        output_names = ['output_seqs', 'p_choose']
        # output_names = ['output_seqs', 'p_choose']
        dynamic_axes_1 = {
            'input_seqs' : {1: 'seq_len'},
            'encoder_output': {1: 'seq_len'},
            'output_seqs' : {1: 'seq_len'},
            'p_choose': {1: 'seq_len', 2: 'dim2'}

        }
        x = (torch.randn((1, 1, 1024),requires_grad=False).float(), torch.randn((1, 16, 1024),requires_grad=False).float(), [4096, 3, {}])
        # torch.save(monotonic_decoder_model.text_decoder_frontend.state_dict(), 'seamless_streaming_monotonic_decoder_text_decoder_frontend.pt')
        torch.onnx.export(monotonic_decoder_model.text_decoder.cpu(), x, 'seamless_streaming_monotonic_decoder_text_decoder.onnx', input_names=input_names, output_names=output_names, verbose='True', opset_version=14, dynamic_axes=dynamic_axes_1)
        """
        """
        print(type(monotonic_decoder_model.final_proj))
        # text_decoder_frontend
        input_names = ['input']
        output_names = ['output']
        dynamic_axes_1 = {
            'input' : {1: 'seq_len'},
            'output' : {1: 'seq_len'}
        }
        x = (torch.randn(1, 2, 1024,requires_grad=False))
        # torch.save(monotonic_decoder_model.text_decoder_frontend.state_dict(), 'seamless_streaming_monotonic_decoder_text_decoder_frontend.pt')
        torch.onnx.export(monotonic_decoder_model.final_proj.cpu(), x, 'seamless_streaming_monotonic_decoder_final_proj.onnx', input_names=input_names, output_names=output_names, verbose='True', opset_version=12, dynamic_axes=dynamic_axes_1)
        """

        return {
            "unity_model": unity_model,
            "unity_config": unity_config,
            "monotonic_decoder_model": monotonic_decoder_model,
            "monotonic_decoder_config": monotonic_decoder_config,
            "text_tokenizer": text_tokenizer,
            "unit_tokenizer": unit_tokenizer,
        }


class UnitYAgentPipeline(UnitYPipelineMixin, AgentPipeline):  # type: ignore
    pipeline: List[GenericAgent] = []

    def __init__(self, args: Namespace):
        models_and_configs = self.load_model(args)

        module_list = []
        for p in self.pipeline:
            module_list.append(
                p.from_args(
                    args,
                    **models_and_configs,
                )
            )
        print(module_list)
        super().__init__(module_list)

    def pop(self, states: Optional[List[Optional[AgentStates]]] = None) -> Segment:
        output_segment = super().pop(states)
        if states is None:
            # Not stateless
            first_states = self.module_list[0].states
        else:
            assert len(states) == len(self.module_list)
            first_states = states[0]

        if not first_states.source_finished and output_segment.finished:
            # An early stop.
            # The temporary solution is to start over
            if states is not None:
                maybe_reset_states(states)
            else:
                self.reset()
            output_segment.finished = False

        return output_segment

    @classmethod
    def from_args(cls, args: Any) -> UnitYAgentPipeline:
        return cls(args)


class UnitYAgentTreePipeline(UnitYPipelineMixin, TreeAgentPipeline):  # type: ignore
    pipeline: Any = {}

    def __init__(self, args: Namespace):
        models_and_configs = self.load_model(args)

        assert len(self.pipeline) > 0
        module_dict = {}
        for module_class, children in self.pipeline.items():
            module_dict[module_class.from_args(args, **models_and_configs)] = children

        super().__init__(module_dict, args)

    @classmethod
    def from_args(cls, args: Any) -> UnitYAgentPipeline:
        return cls(args)

    def pop(
        self, states: Optional[List[Optional[AgentStates]]] = None
    ) -> List[Segment]:
        output_segment = super().pop(states)
        if states is None:
            # Not stateless
            first_states = self.source_module.states
        else:
            assert len(states) == len(self.module_dict)
            first_states = states[self.source_module]

        if isinstance(output_segment, list):
            finished = any(segment.finished for segment in output_segment)
        else:
            # case when output_index is used
            finished = output_segment.finished
        if not first_states.source_finished and finished:
            # An early stop.
            # The temporary solution is to start over
            if states is not None:
                maybe_reset_states(states)
            else:
                self.reset()
            if isinstance(output_segment, list):
                for segment in output_segment:
                    segment.finished = False
            else:
                output_segment.finished = False

        return output_segment  # type: ignore[no-any-return]
