from setuptools import setup, find_packages

setup(
    name="scLinformer",                     
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.10",
    include_package_data=True,
    package_data={
        "scLinformer": ["default_gene_vocab.json"],
    },
    install_requires=[
        "numpy>=1.23,<2.1",
        "scipy>=1.9",
        "pandas>=1.5",
        "scikit-learn>=1.2",
        "torch>=2.0",
        "anndata>=0.9",
        "scanpy>=1.9",
        "matplotlib>=3.6",
        "seaborn>=0.12",
        "adjustText>=0.8",
        "tqdm",
        "leidenalg",
        "igraph"
    ],
)
