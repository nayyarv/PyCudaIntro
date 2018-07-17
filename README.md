# PyCudaIntro

This short talk is about the idea of numerical optimisation in python, using GMMs as an illuminating example and CUDA as the main solution. This introduces the basic concepts, terminology and syntax used in CUDA, as well as testing validity and time calculations.

This doesn't look into CUDA too deeply (for fears of it gazing back :P), or how PyCUDA works, but rather should be enough for you to get basic CUDA acceleration working for your Python Code. 

This computational aspect was part of a thesis into emotion recognition using Markov Chain Monte Carlo. (Hence the 13 dimensional data (MFCCs) and 1 million likelihood evaluations). Code has had a significant facelift since being first written.

The equation being optimised is the [log likelihood](Talk/pressi.pdf) on slide 4.

Presented at July's Sydney Python meet.

## Source

- likelihood/
	- [simple.py](source/likelihood/simple.py) - an inefficient pure python/numpy implementation written by me before realising computational time was a thing
	- [scikitLL.py](source/likelihood/scikitLL.py) - an implementation using scikit-learn's GMM code. Originally used the `sklearn.mixture.GMM` class but this has been deprecated and significantly sped up with the `sklearn.mixture.gaussian_mixture.GaussianMixture` class. Uses scipy and I suspect compiled fortran/C subroutes under the hood
	- [cudaLL.py](source/likelihood/cudaLL.py) - an implementation using `PyCUDA` and the my kernel as in [kernel.cu](source/likelihood/kernel.cu).  The heavy lifting is in the .cu file and the python file is just setting everything up.
	- base.py - the object to mock/subclass for interchangeable use
	- tests.py - some quick checks
- timeRunner.py - runs the various implenetations with various powers of 10. Use `--method` to choose the method to check
- test_validity - a pytest file to compare the results of the output using randomized input

## Talk

Includes the .tex file and the .pdf file as compiled. Not a great complete reference, but a decent starting point.

## Notebooks

Really not necessary, but included anyway

# Links

- [PyCUDA](https://documen.tician.de/pycuda/) - the package used to abstract away the CUDA interface. Documentation is good enough, though a bit sparse and code itself doesn't have much documentation, which makes advanced usage tricky. Has a small section on metaprogramming CUDA too which is very interesting.
- [CUDA Intro from NVIDIA](https://devblogs.nvidia.com/even-easier-introduction-cuda/) - very good intro to CUDA using pure C++ and full of useful links to more advanced understanding. 
- [scikit-learn](https://github.com/scikit-learn/scikit-learn) - a package with implentations of ML and used for their efficient code. Specific page on [GMMs](http://scikit-learn.org/stable/modules/mixture.html) gives more background. Code is very well documented allowing relatively easy use in other applications.
