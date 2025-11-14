from setuptools import setup, find_packages

setup(
    name="hierarchical-mil-transformer",
    version="0.1.0",
    description="Hierarchical MIL Transformer for Whole Slide Image Analysis",
    packages=find_packages(),
    install_requires=[
        "torch>=1.12.0",
        "torchvision>=0.13.0",
        "numpy>=1.21.0",
        "Pillow>=8.3.0",
        "openslide-python>=1.1.2",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.5.0",
        "tqdm>=4.62.0",
        "pandas>=1.3.0",
        "opencv-python>=4.5.0",
        "timm>=0.6.0",
        "transformers>=4.20.0",
        "einops>=0.4.0",
        "h5py>=3.7.0",
        "tensorboard>=2.9.0",
        "albumentations>=1.2.0",
    ],
    python_requires=">=3.8",
)