# CLIP_Paddle
A PaddlePaddle version implementation of CLIP of OpenAI.[【origin repo】](https://github.com/openai/CLIP/)

## Quick Start
```python
import paddle
from PIL import Image
from clip import clip_vit_b_32, tokenize, transforms

model = clip_vit_b_32()
params = paddle.load([path to pretrained model])
model.set_dict(params)
model.eval()

image = transforms(Image.open("CLIP.png")).unsqueeze(0)
text = tokenize(["a diagram", "a dog", "a cat"])

with paddle.no_grad():
    logits_per_image, logits_per_text = model(image, text)
    probs = paddle.nn.functional.softmax(logits_per_image, axis=-1)

print(probs.numpy())
```
    [[0.9927937  0.00421065 0.00299568]]
## Pretrained Models
* [RN50](https://bj.bcebos.com/v1/ai-studio-online/6ffc89246e974a809e6e4b40fdb58063a112a0153e674dae8ed5b6dfe5d46d86?responseContentDisposition=attachment%3B%20filename%3DRN50.pdparams)
* [RN50*4](https://bj.bcebos.com/v1/ai-studio-online/6ffc89246e974a809e6e4b40fdb58063a112a0153e674dae8ed5b6dfe5d46d86?responseContentDisposition=attachment%3B%20filename%3DRN50x4.pdparams)
* [RN101](https://bj.bcebos.com/v1/ai-studio-online/6ffc89246e974a809e6e4b40fdb58063a112a0153e674dae8ed5b6dfe5d46d86?responseContentDisposition=attachment%3B%20filename%3DRN101.pdparams)
* [VIT-B-32](https://bj.bcebos.com/v1/ai-studio-online/eb5e4dbf1ec142caa003a27cefd510ef46a8a6c3932a4d60bfecb3f3ab746c02?responseContentDisposition=attachment%3B%20filename%3DViT-B-32.pdparams)
