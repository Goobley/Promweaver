from setuptools import setup


def readme():
    with open("README.md", "r") as f:
        return f.read()


setup(
    name="promweaver",
    setup_requires=["setuptools_scm"],
    use_scm_version=True,
    packages=["promweaver"],
    install_requires=["lightweaver>=0.8.0rc6"],
    author="Chris Osborne",
    author_email="christopher.osborne@glasgow.ac.uk",
    license="MIT",
    url="http://github.com/Goobley/Promweaver",
    description="Prominence/Filament Radiative Transfer Modelling with Lightweaver",
    long_description=readme(),
    long_description_content_type="text/markdown",
    python_requires=">=3.8",
)
