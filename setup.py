from setuptools import setup, find_packages

setup(
    name="desktop-llm",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "fastapi>=0.104.1",
        "pydantic>=2.5.2",
        "httpx>=0.25.2",
        "torch",
        "torchaudio",
        "transformers",
        "librosa",
        "soundfile",
        "scipy",
        "numpy",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.3",
            "pytest-asyncio>=0.21.1",
            "pytest-cov>=4.1.0",
            "pytest-xdist",
            "black>=23.11.0",
            "isort>=5.12.0",
        ],
    },
)
