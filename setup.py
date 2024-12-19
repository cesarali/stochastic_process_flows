from setuptools import setup, find_packages

install_requires = [
    "pytest==8.3.4",
    "scipy==1.14.1",
    "pytorch-lightning==2.4.0",
    "torchdiffeq==0.2.5",
    "torchsde==0.2.6",
    "matplotlib==3.9.3",
    "wget==3.2",
    "torchtyping==0.1.5",
    "optuna==4.1.0",
    "torchcde==0.2.5",
    "orjson==3.10.12",
    "mlflow==2.19.0"
]

with open('README.md', 'r') as f:
    long_description = f.read()

setup(name='spflows',
      version='0.1.0',
      description='Time series Flow Matching',
      long_description=long_description,
      long_description_content_type='text/markdown',
      url='',
      author='César Ojeda',
      author_email='cesarali07@gmail.com', 
      packages=find_packages(),
      install_requires=install_requires,
      python_requires='>=3.11.9',
      zip_safe=False,
)
