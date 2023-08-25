from typing import Optional
import torch
from hear_ced.hear_wrapper import HearEvalWrapper


def load_model(
    model_file_path: Optional[str] = None,
    model_name="ced_base",
    device: Optional[str] = None,
    **kwargs,
) -> HearEvalWrapper:
    if device is None:
        torch_device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
    else:
        torch_device = torch.device(device)

    # Instantiate model
    model = HearEvalWrapper(model_name, **kwargs)
    model = model.to(torch_device).eval()

    # Set model weights using checkpoint file
    if model_file_path is not None:
        checkpoint = torch.load(model_file_path, map_location=device)
        if 'model' in checkpoint:
            checkpoint = checkpoint['model']
        model.model_impl.load_state_dict(checkpoint, strict=False)

    model.sample_rate = 16000  # Input sample rate
    model.scene_embedding_size = model.embed_dim
    model.timestamp_embedding_size = model.embed_dim
    return model


def get_scene_embeddings(x: torch.Tensor, model: HearEvalWrapper):
    model.eval()
    with torch.no_grad():
        embeddings = model.clip_embedding(x)
    return embeddings


def get_timestamp_embeddings(x: torch.Tensor, model: HearEvalWrapper):
    model.eval()
    with torch.no_grad():
        time_output, time_stamps = model.segment_embedding(x)
    return time_output, time_stamps
