from setuptools import setup

setup(
    name="happyhorse-1-api",
    version="0.1.0",
    author="Anil Matcha",
    description="A comprehensive Python wrapper for the HappyHorse 1.0 API by Alibaba.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    py_modules=["happyhorse_api", "mcp_server"],
    install_requires=[
        "requests",
        "python-dotenv",
        "mcp[cli]"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
