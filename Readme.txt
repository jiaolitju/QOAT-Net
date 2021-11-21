System Requirements

Hardware requirements

QOAT-Net requires a standard windows computer and some common hardware configurations.

Software requirements
OS Requirements
This net is supported for Windows. The net has been tested on the following systems:
	Windows7: 6.1.7601.24535
	Windows10: 10.0.17763.1697

Python Dependencies
QOAT-Net mainly depends on the Python scientific stack.

tensorflow
numpy
scipy
matplotlib

Installation Guide:

Install Anaconda
Anaconda is an open source tool that contains more than 180 science packages and their dependencies, including Conda, Python, numpy and scipy.
Anaconda version is Anaconda3-4.2.0-Windows-x86_64. You can find the version you want on the anaconda official website: https://www.anaconda.com/products/individual#macos. It will probably take about 7 minutes to set up.

Install Pycharm
PyCharm is a Python IDE with a full set of tools to help users to improve their productivity while developing in the Python language, such as debugging, syntax highlighting, Project management, code skipping, smart tips, auto-completion, unit testing, and version control.
The pycharm version is pycharm-community-2018.1. The community version is open source and can be downloaded from the website: https://www.jetbrains.com/pycharm/download/#section=windows. It will take about 5 minutes to set up.  

Install Tensorflow
Tensorflow is a symbolic mathematics system based on dataflow programming, which is widely used in all kinds of machine learning algorithm.
On Windows system, open the command prompt and input: 

conda install tensorflow        
Wait about 3 minutes for installation to complete.                                                                                                                                                                                                                                                                                                                                              

Setting up the development environment:  
Click the PyCharm icon to run the software and create a new project. Then, the environment of the project is configured, and the configuration path is as follows:
File--Setting--Project--Project Interpreter--Set--Add--System Interpreter; and finally select the path where the Anaconda installed.

If you configure the PyCharm environment for the first time, it will take about 10 minutes to initialize the Python environment. 
