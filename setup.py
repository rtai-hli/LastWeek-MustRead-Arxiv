from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements/requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="lastweek-mustread-arxiv",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="An intelligent arXiv paper analysis and newsletter generation system",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/LastWeek-MustRead-Arxiv",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=8.1.1",
            "pytest-cov>=4.1.0",
            "black>=24.2.0",
            "isort>=5.13.2",
            "flake8>=7.0.0",
            "mypy>=1.9.0",
        ],
    },
) 