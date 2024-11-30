from setuptools import setup, find_packages

setup(
    name='flood_prediction',
    version='1.0.0',
    packages=find_packages(),
    install_requires=[
        'flask', 'pandas', 'scikit-learn', 'folium', 'matplotlib'
    ]
)
