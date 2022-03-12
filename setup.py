from setuptools import setup, find_packages

setup(
  name = 'RQ-transformer',
  packages = find_packages(exclude=[]),
  version = '0.0.4',
  license='MIT',
  description = 'RQ Transformer - Autoregressive Transformer for Residual Quantized Codes',
  author = 'Phil Wang',
  author_email = 'lucidrains@gmail.com',
  url = 'https://github.com/lucidrains/RQ-transformer',
  keywords = [
    'artificial intelligence',
    'deep learning',
    'transformers',
    'attention-mechanism',
    'autoregressive',
  ],
  install_requires=[
    'einops>=0.4',
    'torch>=1.6',
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
