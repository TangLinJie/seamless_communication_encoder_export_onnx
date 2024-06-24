# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# MIT_LICENSE file in the root directory of this source tree.

from typing import Optional, Tuple, final

from fairseq2.models.transformer.frontend import TransformerFrontend
from fairseq2.nn.incremental_state import IncrementalStateBag
from fairseq2.nn.padding import PaddingMask
from fairseq2.nn.projection import Projection
from overrides import final as finaloverride
from torch import Tensor
from torch.nn import Module

from seamless_communication.models.monotonic_decoder.monotonic_decoder import (
    MonotonicTransformerDecoder,
)


@final
class MonotonicDecoderModel(Module):
    text_decoder_frontend: TransformerFrontend
    text_decoder: MonotonicTransformerDecoder
    final_proj: Projection

    def __init__(
        self,
        text_decoder_frontend: TransformerFrontend,
        text_decoder: MonotonicTransformerDecoder,
        final_proj: Projection,
    ) -> None:
        super().__init__()

        self.text_decoder_frontend = text_decoder_frontend
        self.text_decoder = text_decoder
        self.final_proj = final_proj
        print(type(text_decoder_frontend))
        print(text_decoder_frontend.embed.embedding_dim)
        # import torch
        # input_names = ['input']
        # output_names = ['output']
        # x = (torch.zeros(1,50, 1024,requires_grad=False).int()) # , torch.ones(50, requires_grad=True).cuda(), torch.randn(1,50, 80,requires_grad=True).cuda(), torch.ones(1,requires_grad=True).cuda()))
        # torch.onnx.export(text_decoder_frontend.cpu().eval(), x, 'seamless_streaming_monotonic_decoder_text_decoder_frontend.onnx', input_names=input_names, output_names=output_names, verbose='True', opset_version=12)

    @finaloverride
    def decode(
        self,
        seqs: Tensor,
        padding_mask: Optional[PaddingMask],
        encoder_output: Tensor,
        encoder_padding_mask: Optional[PaddingMask],
        *,
        state_bag: Optional[IncrementalStateBag] = None,
    ) -> Tuple[Tensor, Optional[PaddingMask], Tensor]:
        # torch.onnx.export(monotonic_decoder_model, x, 'seamless_streaming_monotonic_decoder.onnx', input_names=input_names, output_names=output_names, verbose='True', opset_version=12)
        # seqs, padding_mask = self.text_decoder_frontend(
        #     seqs, padding_mask, state_bag=state_bag
        # )
        if padding_mask is not None:
            padding_mask_params = [padding_mask.seq_lens, padding_mask.batch_seq_len, padding_mask.materialized]
        else:
            padding_mask_params = None
        if state_bag is not None:
            state_bag_params = [state_bag.max_num_steps, state_bag.step_nr, state_bag._module_states]
        else:
            state_bag_params = None
        print('Monotonic text_decoder_frontend input seqs.shape: ', seqs.shape)
        print('Monotonic text_decoder_frontend input seqs: ', seqs)
        print('Monotonic text_decoder_frontend input padding_mask_params: ', padding_mask_params)
        print('Monotonic text_decoder_frontend input state_bag_params: ', state_bag_params)
        # import numpy as np
        # np.savez('input/input.npz',seqs=seqs.detach().numpy(), padding_mask_params=padding_mask_params, state_bag=state_bag_params)
        seqs, padding_mask_params = self.text_decoder_frontend(
            seqs, padding_mask_params, state_bag_params
        )
        # import numpy as np
        # np.savez('output/output.npz',embeds=seqs.detach().numpy())
        print('Monotonic text_decoder_frontend output seqs.shape: ', seqs.shape)
        print('Monotonic text_decoder_frontend output padding_mask_params: ', padding_mask_params)

        if encoder_padding_mask is not None:
            encoder_padding_mask_params = [encoder_padding_mask.seq_lens, encoder_padding_mask.batch_seq_len, encoder_padding_mask.materialized]
        else:
            encoder_padding_mask_params = None
        print('Monotonic text_decoder input seqs.shape: ', seqs.shape)
        print('Monotonic text_decoder input seqs: ', seqs)
        print('Monotonic text_decoder input padding_mask_params: ', padding_mask_params)
        print('Monotonic text_decoder input encoder_output.shape: ', encoder_output.shape)
        print('Monotonic text_decoder input encoder_output: ', encoder_output)
        print('Monotonic text_decoder input encoder_padding_mask_params: ', encoder_padding_mask_params)
        print('Monotonic text_decoder input state_bag_params: ', state_bag_params)
        # assert(len(encoder_output.shape) == 3, encoder_output)
        # import numpy as np
        # np.savez('input/input.npz',input_seqs=seqs.detach().numpy(), input_padding_mask_params=None, encoder_output=encoder_output, encoder_padding_mask_params=None, state_bag=state_bag_params)
        return self.text_decoder(  # type: ignore[no-any-return]
            seqs,
            padding_mask_params,
            encoder_output,
            encoder_padding_mask_params,
            state_bag_params,
        )

    @finaloverride
    def project(self, decoder_output: Tensor) -> Tensor:
        print('Monotonic final_proj input decoder_output.shape: ', decoder_output.shape)
        print('Monotonic final_proj input decoder_output: ', decoder_output)
        # import numpy as np
        # np.savez('input/input.npz',input=decoder_output.detach().numpy())
        logits = self.final_proj(decoder_output)

        # np.savez('output/output.npz',output=logits.detach().numpy())
        print('Monotonic final_proj output logits.shape: ', logits.shape)
        print('Monotonic final_proj output logits: ', logits)
        return logits  # type: ignore[no-any-return]
