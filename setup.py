from setuptools import setup, find_packages

setup(
    name='my_project',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'torch',
        'torchvision',
        'pytorch-lightning',
        'hydra-core',
        'omegaconf',
        'scikit-learn',
        'matplotlib',
        'tqdm'
    ],
)
