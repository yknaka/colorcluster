import setuptools
with open("README.md", "r", encoding="utf-8") as fh:
  long_description = fh.read()


def _requires_from_file(filename):
  return open(filename).read().splitlines()


setuptools.setup(
    name="colorcluster",
    version="1.2.1",
    author="Yuki NAKAMURA",
    author_email="naka_yk@live.jp",
    description="Reduce color of the picture",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yknaka/colorcluster",
    project_urls={
        "Reduce color of the picture": "https://github.com/yknaka/colorcluster",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    py_modules=['colorcluster'],
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
    install_requires=_requires_from_file('requirements.txt'),
    entry_points={
        'console_scripts': [
            'colorcluster = colorcluster:main'
        ]
    },
)
