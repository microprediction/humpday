import pathlib
from setuptools import setup, find_packages

HERE = pathlib.Path(__file__).parent

README = (HERE / "README.md").read_text()

setup(
    name="humpday",
    version="0.7.1",
    description="Taking the pain out of choosing a Python global optimizer",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/microprediction/humpday",
    author="microprediction",
    author_email="pcotton@intechinvestments.com",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
    ],
    packages=["humpday", "humpday.objectives", "humpday.inclusion","humpday.optimizers", "humpday.comparison", "humpday.visualization","humpday.transforms"],
    test_suite='pytest',
    tests_require=['pytest'],
    include_package_data=True,
    extras_require = {
        'full':  ["bayesian-optimization>=1.2.0","optuna","platypus-opt","poap","pysot","ultraopt","dlib",
                  "ax-platform","py-bobyqa","hebo","nlopt","freelunch"]
    },
    install_requires=["wheel","numpy>=1.21.5", "importlib-metadata>=1.7.0", "getjson",
                      "scipy", "funcy", "scipy","winning>=0.4.6","microconventions>=0.5.0",
                       "landscapes",  "pymorton","scikit-learn"],
    entry_points={
        "console_scripts": [
            "humpday=humpday.__main__:main",
        ]
    },
)
