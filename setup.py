from setuptools import find_packages, setup

setup(
    name="scrag",
    version="0.2.0",
    description="Production-grade Legal RAG Pipeline for Supreme Court of India judgments",
    packages=find_packages(include=["scrag", "scrag.*"]),
    python_requires=">=3.10",
)
