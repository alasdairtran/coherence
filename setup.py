from setuptools import setup

setup(name='coherence',
      version='0.1',
      description='Text coherence experiments',
      url='https://github.com/alasdairtran/coherence',
      author='Alasdair Tran',
      author_email='alasdair.tran@anu.edu.au',
      license='MIT',
      packages=['coherence'],
      install_requires=[
          'allennlp',
      ],
      zip_safe=False)
