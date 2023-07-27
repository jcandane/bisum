from distutils.core import setup

with open("README.md") as readme:
    long_description = readme.read()

setup(
  name = 'bisum',         # How you named your package folder (MyLib)
  packages = ['bisum'],   # Chose the same as "name"
  version = '0.1.1',      # Start with a small number and increase it with every change you make
  license='MIT',        # Chose a license from here: https://help.github.com/articles/licensing-a-repository
  description = "binary sparse and dense tensor partial-tracing",   # Give a short description about your library
  long_description=long_description,
  long_description_content_type="text/markdown",
  author = 'Julio Candanedo',                   # Type in your name
  author_email = 'juliojcandanedo@gmail.com',      # Type in your E-Mail
  url = 'https://github.com/jcandane/bisum/',   # Provide either the link to your github or to your website
  keywords = ['pytorch', 'torch', 'tensors', 'Sparse Tensor', 'Sparse', 'contraction', 'partial-tracing', 'einsum', 'tensordot','attention'],   # Keywords that define your package best
  install_requires=[            # I get to this in a second
          'torch'
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
    'Intended Audience :: Developers',      # Define that your audience are developers
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',   # Again, pick a license
    'Programming Language :: Python :: 3',
  ],
)