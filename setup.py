import sys
from setuptools import setup, find_packages

sys.path[0:0] = ['Calib-Challange']
#from version import __version__

setup(
  name = 'Calib-Challange',
  packages = find_packages(),
  #version = __version__,
  license='MIT',
  description = 'Predicting YAW and PITCH from 1 min videos of people driving',
  author = 'Shauray Singh',
  author_email = 'shauray9@gmail.com',
  url = 'https://github.com/shauray8/Calib-Challange',
  keywords = ['deep learning',"comma ai","self driving", 'machine learning'],
  install_requires=[
      'numpy',
      'tqdm',
      'torch',
      'torchvision',
      'pillow',
      "opencv2",
      "tensorboard".

  ],
)
