import numpy as np
import onnxruntime as ort
import sys

from cv2 import imread, imwrite, IMREAD_UNCHANGED, IMWRITE_TIFF_COMPRESSION
from cv2.typing import MatLike
from gc import collect
from transformers import Swin2SRImageProcessor
from logging import basicConfig, getLogger, StreamHandler, INFO


T = -0.51082562376  # ln(0.6)

basicConfig(level=INFO)
logger = getLogger('astro-swin')
handler = StreamHandler(sys.stdout)
logger.addHandler(handler)


def autostretch(image: np.ndarray, eps: float = 1e-2):
    im_min, im_max = image.min(), image.max()
    min_max_scaled = (image - im_min) / (im_max - im_min)
    mean = min_max_scaled.mean()
    mean_scaled = ((1+eps)/(mean+eps)) * mean
    best_gamma =  T / np.log(mean_scaled)
    scale = (1 + eps) / (min_max_scaled + eps)
    scaled_image = (min_max_scaled * scale) ** best_gamma
    return {
        'img': scaled_image,
        'scale': scale,
        'gamma': best_gamma,
        'im_max': im_max,
        'im_min': im_min,
    }

def unstretch(img: np.ndarray, scale: np.ndarray, gamma: float, im_max: float, im_min: float) -> np.ndarray:
    return ((img ** (1/gamma)) / scale) * (im_max - im_min) + im_min

def to_pil(img: np.ndarray):
    img = np.clip(img, 0, 1)
    img = np.moveaxis(img, source=0, destination=-1)
    return img.astype(np.float32)

def get_pad_based_size(size: int, window: int, pad: int):
    return (size // (window - pad) + int(size % (window - pad) != 0)) * (window - pad) + pad

def create_weight_mask(size, overlap):
    mask = np.ones((1, 1, size, size))
    fade = np.linspace(0, 1, overlap)
    # horizontal
    mask[..., :overlap, :] *= fade.reshape(1, 1, -1, 1)
    mask[..., -overlap:, :] *= np.flip(fade, axis=0).reshape(1, 1, -1, 1)
    # vertical
    mask[..., :, :overlap] *= fade.reshape(1, 1, 1, -1)
    mask[..., :, -overlap:] *= np.flip(fade, axis=0).reshape(1, 1, 1, -1)
    return mask

def terminate_blur(
    image: MatLike,
    session: ort.InferenceSession,
    processor: Swin2SRImageProcessor,
):
    window = 256
    pad = 32
    pad_based_width, pad_based_height = get_pad_based_size(image.shape[1], window, pad), get_pad_based_size(image.shape[0], window, pad)

    img_tensor = processor(image, do_rescale=False, return_tensors='np').pixel_values
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
        logger.info(f'{progress * 100:0.2f}%:\t[{"|" * int(progress * 10) + " " * int((1 - progress) * 10)}]')
    target /= np.clip(weight_sum, a_min=1e-6, a_max=weight_sum.max())
    return to_pil(target[0][:, :image.shape[0], :image.shape[1]])

def process(
    model_path: str,
    image_input_path: str,
    image_output_path: str,
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
    image = imread(image_input_path, IMREAD_UNCHANGED)
    stretch_result = autostretch(image)

    stretch_result['img'] = terminate_blur(stretch_result['img'], sess, processor)
    linear_tensor = unstretch(**stretch_result)
    imwrite(image_output_path, linear_tensor, [IMWRITE_TIFF_COMPRESSION, 0])

def parse_args(arg_list: list[str]):
    assert len(arg_list) % 2 == 0

    ARGNAME_TO_PARAM = {
        '-i': 'image_input_path',
        '-o': 'image_output_path',
        '-m': 'model_path',
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
