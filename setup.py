import setuptools

setuptools.setup(
    name="nanograd",
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
)
