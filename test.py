import torch
from hear_ced import ced_base, ced_mini, ced_tiny, ced_small

SR = 16000
duration = 10
audio = torch.randn((1, SR * duration))
mdl = ced_tiny.load_model()

embed, time_stamps = ced_base.get_timestamp_embeddings(audio, mdl)
print(embed.shape)
embed = ced_base.get_scene_embeddings(audio, mdl)
print(embed.shape)
