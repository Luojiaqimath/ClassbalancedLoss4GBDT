from setuptools import setup, find_packages

setup(
    name='iblloss',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scikit-learn',
    ],
    author='Jiaqi Luo, Yuan Yuan, Shixin Xu',
    author_email='jiaqi.luo.jqluo@outlook.com',
    description='A package containing class-balance loss functions for gradient boosting decision tree',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/Luojiaqimath/IBLoss4GBDT',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='Python>=3.9,<3.12',
)

