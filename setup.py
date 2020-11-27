import pathlib
from setuptools import setup, find_packages

HERE = pathlib.Path(__file__).parent

README = (HERE / "README.md").read_text()

setup(
    name="humpday",
    version="0.0.2",
    description="To cheer you up on humpday. Global optimization stuff.",
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
    packages=["humpday"],
    test_suite='pytest',
    tests_require=['pytest', 'shgo'],
    include_package_data=True,
    install_requires=["shgo", "wheel", "pathlib","optuna","sklearn","pymoo","deap","embarrassingly"],
    entry_points={
        "console_scripts": [
            "humpday=humpday.__main__:main",
        ]
    },
)
