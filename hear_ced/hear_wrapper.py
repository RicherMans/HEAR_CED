from hear_ced.models.audiotransformer import AudioTransformer
import hear_ced.models as models
import torch
from typing import Optional, Tuple
from einops import rearrange, reduce, repeat


def upsample(x, ratio, target_length):
    x = rearrange(x, 'b t d -> b d t')
    x = repeat(x, '... t -> ... (t r)', r=ratio)
    left_over = target_length - x.shape[-1]
    # Pad leftovoer with reflection, might be only some frames
    if left_over > 0:
        x = torch.nn.functional.pad(x, (0, left_over), 'replicate')
    elif left_over < 0:
        time_len = x.shape[-1]
        startcrop = time_len // 2 - (target_length // 2)
        end_crop = startcrop + target_length
        x = x[..., startcrop:end_crop]
    x = rearrange(x, 'b d t -> b t d')
    return x


# Might use it for other purposes
def overlapping_windows(
        x: torch.Tensor,  # raw wave tensor, ( Batch, Time)
        win_size: int,  # In samples
        hop_size: int,  # In samples
        center: bool = True) -> torch.Tensor:
    if center:
        x = torch.nn.functional.pad(x, (win_size // 2, win_size // 2 - 1),
                                    mode='constant')
    x = rearrange(x, 'batch time -> batch 1 1 time')

    x = torch.nn.functional.unfold(x,
                                   kernel_size=(1, win_size),
                                   stride=(1, hop_size))
    return rearrange(x, 'b t chunks -> chunks b t')


class HearCEDModel(AudioTransformer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **{'eval_avg': 'cat', **kwargs})

    def forward_spectrogram(self, x: torch.Tensor) -> torch.Tensor:
        x = rearrange(x, 'b f t -> b 1 f t')
        if self.init_bn is not None:
            x = self.init_bn(x)
        if x.shape[-1] > self.maximal_allowed_length:
            # When testing with a longer input
            # splits = x.unfold(-1, self.target_length,
            # 16).permute(3, 0, 1, 2, 4)
            # for f in splits:
            # Just drop the last sample, enhances performance
            splits = x.split(self.target_length, -1)

            if splits[-1].shape[-1] < self.target_length:
                if self.pad_last:
                    pad = torch.zeros(*x.shape[:-1],
                                      self.target_length,
                                      device=x.device)
                    pad[..., :splits[-1].shape[-1]] = splits[-1]
                    splits = torch.stack((*splits[:-1], pad), dim=0)
                else:
                    splits = torch.stack(splits[:-1], dim=0)
            else:
                splits = torch.stack(splits[:-1], dim=0)
            n_splits = len(splits)
            x = rearrange(splits, 'spl b c f t-> (spl b) c f t')
            x = self.forward_head(self.forward_features(x))
            x = rearrange(x, '(spl b) ... d -> spl b (...) d', spl=n_splits)
            if self.eval_avg == 'mean':
                x = x.mean(0)
            elif self.eval_avg == 'max':
                x = x.max(0)[0]
            elif self.eval_avg == 'cat':
                x = rearrange(x, 'spl b ... d -> b (spl ..) d')
            else:
                raise ValueError(
                    f'Unknown Eval average function ({self.eval_avg})')

        else:
            x = self.forward_features(x)
            x = self.forward_head(x)
        return x


def frame_audio_einops(audio: torch.Tensor, frame_size: int, hop_size: float,
                       sample_rate: int) -> Tuple[torch.Tensor, torch.Tensor]:

    audio = torch.nn.functional.pad(
        audio, (frame_size // 2, frame_size - frame_size // 2))
    audio = rearrange(audio, 'b t -> b 1 1 t')
    frame_step = int(hop_size / 1000.0 * sample_rate)
    dx = torch.nn.functional.unfold(audio,
                                    kernel_size=(1, frame_size),
                                    stride=(1, frame_step))

    time_stamps = (torch.arange(0, dx.shape[-1]) * hop_size).float()
    return dx, time_stamps


class HearEvalWrapper(torch.nn.Module):
    """A proxy for efficient net models"""

    def __init__(self, model_name, pooling_type='mean', **kwargs) -> None:
        super().__init__()
        self.model_impl = getattr(models,model_name)(pooling='logit',
                                       **kwargs)
        self.model_impl.__class__ = HearCEDModel
        self.model_impl.forward_head = lambda x: x
        self.pooling_type = pooling_type
        self.overlap = kwargs.get('overlap', True)
        if self.overlap:
            self.hop_size_in_ms = kwargs.get('overlap_hop', 50.)
            # One patch of 160ms is 2559 samples
            # We use here as default 5 patches consecutively, i.e., 7677
            self.overlap_frame = kwargs.get('overlap_frame', 7677)
        else:
            self.hop_size_in_ms = 160  # Standard patchsize = 16
        self.embed_dim = self.model_impl.embed_dim
        self.sample_rate = 16_000

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        # A bit overhead but alright
        input_num_frames = self.model_impl.front_end(x).shape[-1]
        # print(x.dtype)
        embed = self.model_impl(x)
        return embed, input_num_frames

    def segment_embedding(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        n_sounds = x.shape[0]
        if self.overlap:
            x, time_steps = frame_audio_einops(x,
                                               frame_size=self.overlap_frame,
                                               hop_size=self.hop_size_in_ms,
                                               sample_rate=self.sample_rate)
            # Repeat for each batch/nsounds
            time_steps = repeat(time_steps, 't -> b t', b=n_sounds)
            x = rearrange(x, 'b winlen nframes -> (b nframes) winlen')
            x, num_input_frames = self.forward(x)
            x = rearrange(x,
                          'b (f t) d -> b f t d',
                          f=self.model_impl.patch_embed.grid_size[0]).mean(1)
            x = x.mean(1, keepdims=True)
            x = rearrange(x, '(b winlen) 1 d -> b winlen d', b=n_sounds)
        else:
            # Can also process in parallel but that might blow up some memory for a very long clip
            # h = self.hop_size_in_ms / 1000. *
            # x = torch.nn.functional.pad(
            # x, (frame_size // 2, frame_size - frame_size // 2))
            x, num_input_frames = self.forward(x)
            x = rearrange(x,
                          'b (f t) d -> b f t d',
                          f=self.model_impl.patch_embed.grid_size[0]).mean(1)
            num_output_tokens = x.shape[1]
            # x = upsample(x, ratio=16, target_length=num_input_frames)

            time_steps = torch.arange(self.hop_size_in_ms // 2,
                                      num_output_tokens * self.hop_size_in_ms,
                                      self.hop_size_in_ms)
            # Repeat for each batch/nsounds
            time_steps = repeat(time_steps, 't -> b t', b=n_sounds)
        return x, time_steps

    def clip_embedding(self, x: torch.Tensor) -> torch.Tensor:
        if self.pooling_type == 'dm':
            segments, _ = self.segment_embedding(x)
            return reduce(segments, 'b t d -> b d', 'mean')
        else:
            segments, _ = self.forward(x)
            return reduce(segments, 'b t d -> b d', 'mean')


if __name__ == "__main__":
    wrapper = HearEvalWrapper('',
                                     overlap=True)
    x = torch.randn(1, 160000)
    print(wrapper.segment_embedding(x)[0].shape)
