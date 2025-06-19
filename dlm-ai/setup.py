from setuptools import setup, find_packages

with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='dynamic-learning-model',
    version='1.5',
    author='Vignesh Thondikulam',
    author_email='vignesh.tho2006@gmail.com',
    description='A Dynamic Learning Model for processing NLP queries using hybrid AI and reasoning.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/VigneshT24/Dynamic_Learning_Model',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'nltk',
        'spacy',
        'better_profanity',
        'word2number'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
        'License :: OSI Approved :: MIT License',
    ],
    python_requires='>=3.7',
    license='MIT',
)
