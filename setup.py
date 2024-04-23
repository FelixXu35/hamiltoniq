import setuptools


setuptools.setup(
    name="hamiltoniq",
    version="0.1.0",
    author="Felix Xu, Louis Chen",
    author_email="xuxt35@outlook.com",
    description="An application-oriented bechmarking toolkit desgined for QPUs.",
    url="https://github.com/FelixXu35/hamiltoniq",
    packages=setuptools.find_packages(),
    include_package_data=True,
    install_requires=[
        "numpy",
        "scipy",
        "qiskit",
        "qiskit-algorithms",
        "qiskit-ibm-runtime",
        "matplotlib",
        "pandas",
        "seaborn",
        "qiskit_aer"
    ],
    extras_require={
        "qiskit": ["qiskit"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
