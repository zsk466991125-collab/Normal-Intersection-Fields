from pathlib import Path
from setuptools import find_packages, setup

ROOT = Path(__file__).parent
setup(
    name='NIF', version='0.1.0',
    description='Normal-Intersection Fields for Oriented Object Detection',
    author='Shaokai Zheng', author_email='zsk@zc-sh.net',
    packages=find_packages(exclude=('third_party', 'third_party.*')),
    install_requires=(ROOT/'requirements.txt').read_text().splitlines(),
    long_description=(ROOT/'README.md').read_text(),
    long_description_content_type='text/markdown',
    python_requires='>=3.9', license='MIT',
)
