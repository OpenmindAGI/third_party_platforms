from setuptools import setup, find_packages

setup(
    name="ubtechapi",                  # the distribution name
    version="0.1.0",
    description="UBTech API client",
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="you@example.com",
    url="https://github.com/yourname/ubtechapi",  # your project URL
    packages=find_packages(include=["ubtechapi", "ubtechapi.*"]),
    include_package_data=True,         # include files from MANIFEST.in, if any
    install_requires=[
        # e.g. "requests>=2.0",
    ],
    python_requires=">=3.6",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
