#!/usr/bin/env python3
"""
Setup script for OperoNN package.

Installation:
    pip install -e .            # Development install (editable)
    pip install .               # Standard install
    pip install .[dev]          # Install with dev dependencies
"""

from setuptools import setup, find_packages
import os

# Read version from package
def get_version():
    version_file = os.path.join(os.path.dirname(__file__), 'operonn', '__init__.py')
    with open(version_file, 'r') as f:
        for line in f:
            if line.startswith('__version__'):
                return line.split('=')[1].strip().strip('"').strip("'")
    return '1.0.0'

# Read long description from README
def get_long_description():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return ''

setup(
    name='operonn',
    version=get_version(),
    author='Abdullah Tariq Choudhry',
    author_email='a_choudhry@u.pacific.edu',
    description='A Neural Network Approach to Bacterial Operon Prediction',
    long_description=get_long_description(),
    long_description_content_type='text/markdown',
    url='https://github.com/Abdullah0x0/operonn',
    
    packages=find_packages(),
    include_package_data=True,
    
    # Include trained models
    package_data={
        'operonn': ['../trained_models/**/*'],
    },
    
    python_requires='>=3.8',
    
    install_requires=[
        'numpy>=1.20.0',
        'pandas>=1.3.0',
        'torch>=1.9.0',
        'scikit-learn>=0.24.0',
    ],
    
    extras_require={
        'dev': [
            'pytest>=6.0',
            'pytest-cov>=2.0',
            'black>=21.0',
            'flake8>=3.9',
        ],
    },
    
    entry_points={
        'console_scripts': [
            'operonn=operonn.cli:main',
        ],
    },
    
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    
    keywords='operon prediction deep-learning bioinformatics bacteria rna-seq',
)

