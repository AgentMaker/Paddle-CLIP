# CLIP_Paddle
A PaddlePaddle version implementation of CLIP of OpenAI. [【origin repo】](https://github.com/openai/CLIP/)

## Install Package
* Install by pip：
```shell
$ pip install paddleclip==1.0.0 -i https://pypi.python.org/pypi 
```
* Install by wheel package：[【Releases Packages】](https://github.com/AgentMaker/Paddle-CLIP/releases)

## Requirements
* wget
* ftfy
* regex
* paddlepaddle(cpu/gpu)>=2.0.1

## Quick Start
```python
import paddle
from PIL import Image
from clip import tokenize, load_model

# Load the model
model, transforms = load_model('ViT_B_32', pretrained=True)

# Prepare the inputs
image = transforms(Image.open("CLIP.png")).unsqueeze(0)
text = tokenize(["a diagram", "a dog", "a cat"])

# Calculate features and probability
with paddle.no_grad():
    logits_per_image, logits_per_text = model(image, text)
    probs = paddle.nn.functional.softmax(logits_per_image, axis=-1)
    
# Print the result
print(probs.numpy())
```
    [[0.9927937  0.00421065 0.00299568]]

## Zero-Shot Prediction
```python
import paddle
from clip import tokenize, load_model
from paddle.vision.datasets import Cifar100

# Load the model
model, transforms = load_model('ViT_B_32', pretrained=True)

# Load the dataset
cifar100 = Cifar100(mode='test', backend='pil')
classes = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 
    'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', 
    'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard', 
    'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 
    'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 
    'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 
    'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 
    'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 
    'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm'
]

# Prepare the inputs
image, class_id = cifar100[3637]
image_input = transforms(image).unsqueeze(0)
text_inputs = tokenize(["a photo of a %s" % c for c in classes])

# Calculate features
with paddle.no_grad():
    image_features = model.encode_image(image_input)
    text_features = model.encode_text(text_inputs)

# Pick the top 5 most similar labels for the image
image_features /= image_features.norm(axis=-1, keepdim=True)
text_features /= text_features.norm(axis=-1, keepdim=True)
similarity = (100.0 * image_features @ text_features.t())
similarity = paddle.nn.functional.softmax(similarity, axis=-1)
values, indices = similarity[0].topk(5)

# Print the result
for value, index in zip(values, indices):
    print('%s: %.02f%%' % (classes[index], value*100.))
```
    snake: 65.31%
    turtle: 12.29%
    sweet_pepper: 3.83%
    lizard: 1.88%
    crocodile: 1.75%

## Linear-probe evaluation
```python
import os
import paddle
import numpy as np
from tqdm import tqdm
from paddle.io import DataLoader
from clip import tokenize, load_model
from paddle.vision.datasets import Cifar100
from sklearn.linear_model import LogisticRegression

# Load the model
model, transforms = load_model('ViT_B_32', pretrained=True)

# Load the dataset
train = Cifar100(mode='train', transform=transforms, backend='pil')
test = Cifar100(mode='test', transform=transforms, backend='pil')

# Get features
def get_features(dataset):
    all_features = []
    all_labels = []
    
    with paddle.no_grad():
        for images, labels in tqdm(DataLoader(dataset, batch_size=100)):
            features = model.encode_image(images)
            all_features.append(features)
            all_labels.append(labels)

    return paddle.concat(all_features).numpy(), paddle.concat(all_labels).numpy()

# Calculate the image features
train_features, train_labels = get_features(train)
test_features, test_labels = get_features(test)

# Perform logistic regression
classifier = LogisticRegression(random_state=0, C=0.316, max_iter=1000, verbose=0)
classifier.fit(train_features, train_labels)

# Evaluate using the logistic regression classifier
predictions = classifier.predict(test_features)
accuracy = np.mean((test_labels == predictions).astype(np.float)) * 100.

# Print the result
print(f"Accuracy = {accuracy:.3f}")
```
    Accuracy = 79.900

## Pretrained Models Download
* [RN50](https://bj.bcebos.com/v1/ai-studio-online/6ffc89246e974a809e6e4b40fdb58063a112a0153e674dae8ed5b6dfe5d46d86?responseContentDisposition=attachment%3B%20filename%3DRN50.pdparams)
* [RN50x4](https://bj.bcebos.com/v1/ai-studio-online/9f874e0174da48ffbd7c17e77b1fb278632620a9995e476ba873e334caec9037?responseContentDisposition=attachment%3B%20filename%3DRN50x4.pdparams)
* [RN101](https://bj.bcebos.com/v1/ai-studio-online/484592d98c584785bc8f6f9f7badbf4a9fb7a96f6102470697ed974e8eeee2a9?responseContentDisposition=attachment%3B%20filename%3DRN101.pdparams)
* [ViT_B_32](https://bj.bcebos.com/v1/ai-studio-online/eb5e4dbf1ec142caa003a27cefd510ef46a8a6c3932a4d60bfecb3f3ab746c02?responseContentDisposition=attachment%3B%20filename%3DViT-B-32.pdparams)
