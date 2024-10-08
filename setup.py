from setuptools import setup, find_packages

setup(
    name="qmodel",
    version="2.0.1",
    description="ML system for QATCH Technologies NanoVisQ software.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Paul MacNichol",
    author_email="paulmacnichol@gmail.com",
    url="https://github.com/PaulMacNichol",
    packages=find_packages(),
    install_requires=[
        "numpy<2.0",
        "pandas",
        "xgboost",
        "tsmoothie",
        "hyperopt",
        "scipy",
        "scikit-learn",
        "matplotlib",
        "tensorflow",
        "tqdm",
        "keras",
        "joblib",
        "pillow",
        "imblearn",
        "seaborn",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    entry_points={
        "console_scripts": [
            # Define command-line scripts here, e.g.
            # 'your_command=your_module_name.module:main_function',
        ],
    },
    include_package_data=True,
    zip_safe=False,
    setup_requires=[
        "setuptools>=42",
    ],
    extras_require={
        "dev": [
            "check-manifest",
            "flake8",
        ],
        "docs": [
            "sphinx",
            "sphinx-rtd-theme",
        ],
    },
    project_urls={
        "Bug Tracker": "https://github.com/QATCH-Technologies/QATCH-ML/issues",
        "Source Code": "https://github.com/QATCH-Technologies/QATCH-ML",
    },
    license="MIT",
)
