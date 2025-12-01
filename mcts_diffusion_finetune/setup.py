"""
Setup script for MCTS Diffusion Fine-tune package.
"""

from setuptools import setup, find_packages
import os

# Read README for long description
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return ''

# Read requirements
def read_requirements(filename='requirements.txt'):
    req_path = os.path.join(os.path.dirname(__file__), filename)
    if os.path.exists(req_path):
        with open(req_path, 'r') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return []

setup(
    name='mcts_diffusion_finetune',
    version='0.1.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='MCTS-guided protein design with diffusion models',
    long_description=read_readme(),
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/mcts_diffusion_finetune',
    packages=find_packages(exclude=['tests', 'tests.*', 'docs', 'scripts', 'notebooks']),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    python_requires='>=3.8',
    install_requires=read_requirements(),
    extras_require={
        'dev': [
            'pytest>=7.0.0',
            'pytest-cov>=3.0.0',
            'black>=22.0.0',
            'flake8>=4.0.0',
            'mypy>=0.950',
            'isort>=5.10.0',
        ],
        'visualization': read_requirements('requirements_visualization.txt'),
        'external_models': read_requirements('requirements_lightweight_experts.txt'),
    },
    entry_points={
        'console_scripts': [
            'mcts-inverse-folding=experiments.run_inverse_folding:main',
            'mcts-forward-folding=experiments.run_forward_folding:main',
            'mcts-motif-scaffolding=experiments.run_motif_scaffolding:main',
        ],
    },
    include_package_data=True,
    package_data={
        'mcts_diffusion_finetune': [
            'config/*.yaml',
            'config/*.json',
        ],
    },
    zip_safe=False,
)
