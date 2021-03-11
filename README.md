# CLIP_Paddle
A PaddlePaddle version implementation of CLIP of OpenAI. [【origin repo】](https://github.com/openai/CLIP/)

## Quick Start
```python
import paddle
from PIL import Image
from clip import tokenize, load_model

model, transforms = load_model('ViT_B_32', pretrained=True)

image = transforms(Image.open("CLIP.png")).unsqueeze(0)
text = tokenize(["a diagram", "a dog", "a cat"])

with paddle.no_grad():
    logits_per_image, logits_per_text = model(image, text)
    probs = paddle.nn.functional.softmax(logits_per_image, axis=-1)

print(probs.numpy())
```
    [[0.9927937  0.00421065 0.00299568]]

## Requirements
* wget
* ftfy
* regex
* paddlepaddle(cpu/gpu)>=2.0.1

## Pretrained Models
* [RN50](https://bj.bcebos.com/v1/ai-studio-online/6ffc89246e974a809e6e4b40fdb58063a112a0153e674dae8ed5b6dfe5d46d86?responseContentDisposition=attachment%3B%20filename%3DRN50.pdparams)
* [RN50x4](https://bj.bcebos.com/v1/ai-studio-online/9f874e0174da48ffbd7c17e77b1fb278632620a9995e476ba873e334caec9037?responseContentDisposition=attachment%3B%20filename%3DRN50x4.pdparams)
* [RN101](https://bj.bcebos.com/v1/ai-studio-online/484592d98c584785bc8f6f9f7badbf4a9fb7a96f6102470697ed974e8eeee2a9?responseContentDisposition=attachment%3B%20filename%3DRN101.pdparams)
* [ViT_B_32](https://bj.bcebos.com/v1/ai-studio-online/eb5e4dbf1ec142caa003a27cefd510ef46a8a6c3932a4d60bfecb3f3ab746c02?responseContentDisposition=attachment%3B%20filename%3DViT-B-32.pdparams)