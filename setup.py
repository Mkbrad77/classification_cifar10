from setuptools import setup, find_packages

setup(
    name='cifar10_classification',
    version='0.1.0',
    description='A CIFAR-10 image classification project',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'scikit-learn',
        'matplotlib',
        'seaborn',
        'scipy',
        'joblib'
    ],
    python_requires='>=3.7',
)
