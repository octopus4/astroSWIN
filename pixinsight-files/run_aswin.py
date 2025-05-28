import numpy as np
import onnxruntime as ort
import sys

from gc import collect
from PIL import Image
from transformers import Swin2SRImageProcessor
from typing import Optional
from logging import basicConfig, getLogger, StreamHandler, INFO

basicConfig(level=INFO)
logger = getLogger('astro-swin')
handler = StreamHandler(sys.stdout)
logger.addHandler(handler)


def to_pil(v: np.ndarray):
    v = np.clip(v, 0, 1)
    v = np.moveaxis(v, source=0, destination=-1)
    return (v * 255.0).round().astype(np.uint8)

def get_pad_based_size(size: int, window: int, pad: int):
    return (size // (window - pad) + int(size % (window - pad) != 0)) * (window - pad) + pad

def calculate_patch_weights(image, patch_size=32, beta=0.5):
    c, h, w = image.shape
    h = (h // patch_size + int(h % patch_size != 0)) * patch_size
    w = (w // patch_size + int(w % patch_size != 0)) * patch_size
    canvas = np.zeros((c, h, w))
    canvas[:, :image.shape[-2], :image.shape[-1]] += image
    canvas = canvas.mean(axis=0)

    h_patches = h // patch_size
    w_patches = w // patch_size
    patches = canvas.reshape(h_patches, patch_size, w_patches, patch_size)
    patches = patches.transpose(0, 2, 1, 3)      # (h_p, w_p, ps, ps)

    means = np.mean(patches, axis=(-2, -1))      # (h_p, w_p)
    variances = np.var(patches, axis=(-2, -1))   # (h_p, w_p)

    means_norm = (means - np.min(means) + 1e-8) / (np.max(means) - np.min(means) + 1e-8)
    variances_norm = (variances - np.min(variances) + 1e-8) / (np.max(variances) - np.min(variances) + 1e-8)

    weights = ((1 + beta**2) * means_norm * variances_norm) / (means_norm * (beta ** 2) + variances_norm)

    weights = weights.repeat(patch_size, axis=0).repeat(patch_size, axis=1)
    fit_weights = np.expand_dims(weights[:image.shape[-2], :image.shape[-1]], 0)
    return np.repeat(fit_weights / fit_weights.max(), 3, 0)

def create_weight_mask(size, overlap):
    mask = np.ones((1, 1, size, size))
    fade = np.linspace(0, 1, overlap)

    mask[..., :overlap, :] *= fade.reshape(1, 1, -1, 1)
    mask[..., -overlap:, :] *= np.flip(fade, axis=0).reshape(1, 1, -1, 1)

    mask[..., :, :overlap] *= fade.reshape(1, 1, 1, -1)
    mask[..., :, -overlap:] *= np.flip(fade, axis=0).reshape(1, 1, 1, -1)

    return mask

def terminate_blur(
    image: Image,
    session: ort.InferenceSession,
    processor: Swin2SRImageProcessor,
    mask_patch_size: Optional[int] = None,
    mask_beta: Optional[float] = None,
    mask_const: Optional[float] = None,
    mask_multiplier: Optional[float] = None,
):
    window = 256
    pad = 32
    pad_based_width, pad_based_height = get_pad_based_size(image.width, window, pad), get_pad_based_size(image.height, window, pad)

    img_tensor = processor(image, return_tensors='np').pixel_values
    pad_based_img = np.zeros((1, 3, pad_based_height, pad_based_width))
    pad_based_img[:, :, :img_tensor.shape[-2], :img_tensor.shape[-1]] += img_tensor
    target = np.zeros_like(pad_based_img)
    weight_sum = np.zeros_like(target)
    weight_mask = create_weight_mask(window, pad)

    x = 0
    while x < pad_based_width - pad:
        x_from, x_to = x, x + window
        y = 0
        while y < pad_based_height - pad:
            y_from, y_to = y, y + window
            patch_tensor = pad_based_img[:, :, y_from:y_to, x_from:x_to]
            outputs = session.run(None, {"pixel_values": patch_tensor.astype(np.float32)})[0]
            target[:, :, y_from:y_to, x_from:x_to] += outputs * weight_mask
            weight_sum[:, :, y_from:y_to, x_from:x_to] += weight_mask
            y = y_to - pad
        collect()
        x = x_to - pad
        progress = x / pad_based_width
        logger.info(f'{progress * 100:0.5f}%: [{"|" * int(progress * 10) + " " * int((1 - progress) * 10)}]')
    target /= np.clip(weight_sum, a_min=1e-6, a_max=weight_sum.max())
    processed, base_image = target[0][:, :image.height, :image.width], img_tensor[0][:, :image.height, :image.width]
    if mask_patch_size is None or mask_beta is None or mask_const is None or mask_multiplier is None:
        return Image.fromarray(to_pil(processed))
    patch_weights = calculate_patch_weights(base_image, patch_size=int(mask_patch_size), beta=float(mask_beta))
    rescaled_weights = np.clip((patch_weights + float(mask_const)) * float(mask_multiplier), a_max=1, a_min=0)
    return Image.fromarray(to_pil(rescaled_weights * processed + (1 - rescaled_weights) * base_image))

def process(
    model_path: str,
    image_input_path: str,
    image_output_path: str,
    mask_patch_size: Optional[int] = None,
    mask_beta: Optional[float] = None,
    mask_const: Optional[float] = None,
    mask_multiplier: Optional[float] = None
):
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
    options.intra_op_num_threads = 12
    options.inter_op_num_threads = 12

    providers = [
        'DmlExecutionProvider',
        'CPUExecutionProvider'
    ]

    sess = ort.InferenceSession(model_path + '/model.onnx', sess_options=options, providers=providers)
    processor = Swin2SRImageProcessor.from_pretrained(model_path)
    image = Image.open(image_input_path).convert('RGB')

    processed = terminate_blur(
        image,
        sess,
        processor,
        mask_patch_size=mask_patch_size,
        mask_beta=mask_beta,
        mask_const=mask_const,
        mask_multiplier=mask_multiplier
    )
    with open(image_output_path, 'wb') as f:
        processed.save(f)

def parse_args(arg_list: list[str]):
    assert len(arg_list) % 2 == 0

    ARGNAME_TO_PARAM = {
        '-i': 'image_input_path',
        '-o': 'image_output_path',
        '-m': 'model_path',
        '--patch-size': 'mask_patch_size',
        '--beta': 'mask_beta',
        '--const': 'mask_const',
        '--mul': 'mask_multiplier',
    }
    result = {}
    for i in range(len(arg_list) // 2):
        result[ARGNAME_TO_PARAM[arg_list[2 * i]]] = arg_list[2 * i + 1]
    return result

def main():
    try:
        arg_list = sys.argv[1:]
        kwargs = parse_args(arg_list)
        process(**kwargs)
    except Exception as e:
        logger.error(e)

main()
