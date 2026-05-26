from setuptools import setup, find_packages

setup(
    name="vlm-ensemble",
    version="0.1.0",
    description="Self-Ensembling Vision-Language Models for Chart Data Extraction",
    author="Thomas Berkane, Qianyi Wang, Maimuna S. Majumder",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.37.0",
        "pandas",
        "numpy",
        "pyyaml",
    ],
)
