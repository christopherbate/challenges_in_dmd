from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

setup(
    name='dmdc',  # Required
    version='1.0.0',  # Required        
    long_description_content_type='text/markdown',  # Optional (see note above)
    keywords='sample, setuptools, development',  # Optional    
    package_dir={'': 'src'},  # Optional    
    packages=find_packages(where='src'),  # Required
    python_requires='>=3.6, <4',
    install_requires=["numpy", "scipy", "matplotlib"],  # Optional
    extras_require={  # Optional        
        # 'dev': ['check-manifest'],
        # 'test': ['coverage'],
    },    
    package_data={  # Optional
        # 'sample': ['package_data.dat'],
    },    
    entry_points={
    },
)