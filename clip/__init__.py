import paddle
from .tokenizer import Tokenizer
from .model import CLIP, clip_rn50, clip_rn101, clip_rn50x4, clip_vit_b_32
from paddle.vision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

tokenizer = Tokenizer()
transforms = Compose([
        Resize(224, interpolation='bicubic'),
        CenterCrop(224),
        lambda image: image.convert("RGB"),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
])


def tokenize(texts, context_length=77):
    """
    Returns the tokenized representation of given input string(s)
    Parameters
    ----------
    texts : Union[str, List[str]]
        An input string or a list of input strings to tokenize
    context_length : int
        The context length to use; all CLIP models use 77 as the context length
    Returns
    -------
    A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length]
    """
    if isinstance(texts, str):
        texts = [texts]

    sot_token = tokenizer.encoder["<|startoftext|>"]
    eot_token = tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + tokenizer.encode(text) + [eot_token] for text in texts]
    result = paddle.zeros((len(all_tokens), context_length), dtype='int64')

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
        result[i, :len(tokens)] = paddle.to_tensor(tokens)

    return result