import numpy as np
import onnxruntime as ort
import sys

from gc import collect
from PIL import Image
from transformers import Swin2SRImageProcessor
from logging import basicConfig, getLogger, StreamHandler, INFO

basicConfig(level=INFO)
logger = getLogger('astro-swin')
logger.addHandler(StreamHandler(sys.stdout))


def to_pil(v: np.ndarray):
    v = np.clip(v, 0, 1)
    v = np.moveaxis(v, source=0, destination=-1)
    return (v * 255.0).round().astype(np.uint8)

def get_pad_based_size(size: int, window: int, pad: int):
    return (size // (window - pad) + int(size % (window - pad) != 0)) * (window - pad) + pad

def create_weight_mask(size, overlap):
    mask = np.ones((1, 1, size, size))
    fade = np.linspace(0, 1, overlap)

    # v borders
    mask[..., :overlap, :] *= fade.reshape(1, 1, -1, 1)
    mask[..., -overlap:, :] *= np.flip(fade, axis=0).reshape(1, 1, -1, 1)

    # h borders
    mask[..., :, :overlap] *= fade.reshape(1, 1, 1, -1)
    mask[..., :, -overlap:] *= np.flip(fade, axis=0).reshape(1, 1, 1, -1)
    return mask

def terminate_blur(image: Image, session: ort.InferenceSession, processor: Swin2SRImageProcessor, window: int = 256):
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
        logger.info(f'{(x / pad_based_width) * 100}% finished')
    target /= np.clip(weight_sum, a_min=1e-6, a_max=weight_sum.max())
    processed, base_image = target[:, :, :image.height, :image.width], img_tensor[:, :, :image.height, :image.width]
    target_mask = base_image.copy()
    target_mean, target_var = target_mask.mean(), np.sqrt(target_mask.var())
    img_mask = np.log(1 + np.abs(base_image - target_mean) / target_var)
    img_mask /= img_mask.max()
    blend = (1 - img_mask) * base_image + img_mask * processed
    return Image.fromarray(to_pil(blend[0]))

def process(model_path: str, image_input_path: str, image_output_path: str):
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
    options.intra_op_num_threads = 12  # Потоки для операций (например, матричные умножения)
    options.inter_op_num_threads = 12  # Потоки для параллельных операций

    providers = [
        'DmlExecutionProvider',
        'CPUExecutionProvider'
    ]

    sess = ort.InferenceSession(model_path + '/model.onnx', sess_options=options, providers=providers)
    processor = Swin2SRImageProcessor.from_pretrained(model_path)
    image = Image.open(image_input_path).convert('RGB')

    processed = terminate_blur(image, sess, processor)
    with open(image_output_path, 'wb') as f:
        processed.save(f)


def parse_args(arg_list: list[str]):
    assert len(arg_list) % 2 == 0

    ARGNAME_TO_PARAM = {'-i': 'image_input_path', '-o': 'image_output_path', '-m': 'model_path'}
    result = {}
    for i in range(len(arg_list) // 2):
        result[ARGNAME_TO_PARAM[arg_list[2 * i]]] = arg_list[2 * i + 1]
    return result


def main():
    arg_list = sys.argv[1:]
    kwargs = parse_args(arg_list)
    process(**kwargs)

main()
