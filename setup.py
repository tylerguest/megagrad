import setuptools

setuptools.setup(
    name="megagrad",
    version="0.0.1",
    author="Tyler Guest",
    description="A tiny scalar-valued autograd engine with a small PyTorch-like neural network library on top.",
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    extras_require={
        'test': [
            'pytest>=6.0',
            'pytest-cov>=2.0',
            'torch>=1.8.0'
        ],
        'dev': [
            'pytest>=6.0',
            'pytest-cov>=2.0',
            'torch>=1.8.0',
            'jupyter>=1.0.0',
            'black>=21.0',
            'isort>=5.0'
        ]
    }
)