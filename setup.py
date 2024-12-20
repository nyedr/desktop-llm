from setuptools import setup, find_packages

setup(
    name="desktop-llm",
    version="0.1.0",
    packages=["app"],
    install_requires=[
        "fastapi>=0.104.1",
        "pydantic>=2.5.2",
        "httpx>=0.25.2",
    ],
)
