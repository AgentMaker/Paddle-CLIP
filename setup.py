from setuptools import setup
setup(
    name='paddleclip',
    version='1.0.0',
    author='jm12138',
    author_email='2286040843@qq.com',
    packages=['clip'],
    license='Apache-2.0 License',
    description='Paddle CLIP',
    install_requires=['wget', 'ftfy', 'regex'],
    package_data={'': ['bpe_simple_vocab_16e6.txt.gz']}
)
