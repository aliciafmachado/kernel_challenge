import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="kernel-challenge",
    version="0.0.1",
    author="Alicia Fortes Machado & Agathe Senellart",
    author_email="aliciafortesmachado@gmail.com",
    description="Kernel methods implementation for kaggle challenge.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/aliciafmachado/kernel_challenge",
    project_urls={
        "Bug Tracker": "https://github.com/aliciafmachado/kernel_challenge/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    package_dir={"kernel_challenge": "src"},
    # where="src"
    packages=setuptools.find_packages(),
    python_requires=">=3.9",
    # install_requires=[
    #             'gym',
    #             'matplotlib',
    #             'numpy',
    #             'jax',
    #             'dm-acme',
    #             'chex',
    #             'dm_env',
    #         ],
)

print(setuptools.find_packages())