import os
import setuptools

readme_path = 'README.md' if os.path.exists('README.md') else 'readme.md'
with open(readme_path, 'r', encoding='utf-8') as fp:
    long_description = fp.read()

with open('requirements.txt') as fp:
    requirements = fp.read().splitlines()

setuptools.setup(
    name='multimodel_analysis',      # Package name
    version='0.1.0',                 # Initial version
    author='Uditya Narayan Tiwari',
    author_email='tiwarimerit@gmail.com',
    description='A Python Package for Automatic Multi-Model Analysis (Classification & Regression)',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
    ],
    install_requires=requirements,
    python_requires='>=3.8',
    include_package_data=True,
)