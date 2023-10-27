import setuptools


setuptools.setup(
    name="hamiltoniq",
    version="0.1.0",
    author="Felix Xu, Louis Chen",
    author_email="xuxt35@outlook.com",
    description="A bechmarking toolkit desgined for QAOA performance on real quantum hardwares.",
    #long_description=long_description,
    #long_description_content_type="text/markdown",
    url="https://github.com/FelixXu35/HamilToniQ",
    packages=setuptools.find_packages(),
    include_package_data=True,
    install_requires=["numpy", "scipy"],
    extras_require={
        "qiskit": ["qiskit"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
