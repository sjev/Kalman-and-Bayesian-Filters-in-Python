#!/usr/bin/env python
# coding: utf-8

# <center><h1>Kalman and Bayesian Filters in Python</h1></center>
# <p>
#  <p>
#
# ## Table of Contents
#
# [**Preface**](./00-Preface.ipynb)
#
# Motivation behind writing the book. How to download and read the book. Requirements for IPython Notebook and Python. github links.
#
#
# [**Chapter 1: The g-h Filter**](./01-g-h-filter.ipynb)
#
# Intuitive introduction to the g-h filter, also known as the $\alpha$-$\beta$ Filter, which is a family of filters that includes the Kalman filter. Once you understand this chapter you will understand the concepts behind the Kalman filter.
#
#
# [**Chapter 2: The Discrete Bayes Filter**](./02-Discrete-Bayes.ipynb)
#
# Introduces the discrete Bayes filter. From this you will learn the probabilistic (Bayesian) reasoning that underpins the Kalman filter in an easy to digest form.
#
# [**Chapter 3: Probabilities, Gaussians, and Bayes' Theorem**](./03-Gaussians.ipynb)
#
# Introduces using Gaussians to represent beliefs in the Bayesian sense. Gaussians allow us to implement the algorithms used in the discrete Bayes filter to work in continuous domains.
#
#
# [**Chapter 4: One Dimensional Kalman Filters**](./04-One-Dimensional-Kalman-Filters.ipynb)
#
# Implements a Kalman filter by modifying the discrete Bayes filter to use Gaussians. This is a full featured Kalman filter, albeit only useful for 1D problems.
#
#
# [**Chapter 5: Multivariate Gaussians**](./05-Multivariate-Gaussians.ipynb)
#
# Extends Gaussians to multiple dimensions, and demonstrates how 'triangulation' and hidden variables can vastly improve estimates.
#
# [**Chapter 6: Multivariate Kalman Filter**](./06-Multivariate-Kalman-Filters.ipynb)
#
# We extend the Kalman filter developed in the univariate chapter to the full, generalized filter for linear problems. After reading this you will understand how a Kalman filter works and how to design and implement one for a (linear) problem of your choice.
#
# [**Chapter 7: Kalman Filter Math**](./07-Kalman-Filter-Math.ipynb)
#
# We gotten about as far as we can without forming a strong mathematical foundation. This chapter is optional, especially the first time, but if you intend to write robust, numerically stable filters, or to read the literature, you will need to know the material in this chapter. Some sections will be required to understand the later chapters on nonlinear filtering.
#
#
# [**Chapter 8: Designing Kalman Filters**](./08-Designing-Kalman-Filters.ipynb)
#
# Building on material in Chapters 5 and 6, walks you through the design of several Kalman filters. Only by seeing several different examples can you really grasp all of the theory. Examples are chosen to be realistic, not 'toy' problems to give you a start towards implementing your own filters. Discusses, but does not solve issues like numerical stability.
#
#
# [**Chapter 9: Nonlinear Filtering**](./09-Nonlinear-Filtering.ipynb)
#
# Kalman filters as covered only work for linear problems. Yet the world is nonlinear. Here I introduce the problems that nonlinear systems pose to the filter, and briefly discuss the various algorithms that we will be learning in subsequent chapters.
#
#
# [**Chapter 10: Unscented Kalman Filters**](./10-Unscented-Kalman-Filter.ipynb)
#
# Unscented Kalman filters (UKF) are a recent development in Kalman filter theory. They allow you to filter nonlinear problems without requiring a closed form solution like the Extended Kalman filter requires.
#
# This topic is typically either not mentioned, or glossed over in existing texts, with Extended Kalman filters receiving the bulk of discussion. I put it first because the UKF is much simpler to understand, implement, and the filtering performance is usually as good as or better then the Extended Kalman filter. I always try to implement the UKF first for real world problems, and you should also.
#
#
# [**Chapter 11: Extended Kalman Filters**](./11-Extended-Kalman-Filters.ipynb)
#
# Extended Kalman filters (EKF) are the most common approach to linearizing non-linear problems. A majority of real world Kalman filters are EKFs, so will need to understand this material to understand existing code, papers, talks, etc.
#
#
# [**Chapter 12: Particle Filters**](./12-Particle-Filters.ipynb)
#
# Particle filters uses Monte Carlo techniques to filter data. They easily handle highly nonlinear and non-Gaussian systems, as well as multimodal distributions (tracking multiple objects simultaneously) at the cost of high computational requirements.
#
#
# [**Chapter 13: Smoothing**](./13-Smoothing.ipynb)
#
# Kalman filters are recursive, and thus very suitable for real time filtering. However, they work extremely well for post-processing data. After all, Kalman filters are predictor-correctors, and it is easier to predict the past than the future! We discuss some common approaches.
#
#
# [**Chapter 14: Adaptive Filtering**](./14-Adaptive-Filtering.ipynb)
#
# Kalman filters assume a single process model, but manuevering targets typically need to be described by several different process models. Adaptive filtering uses several techniques to allow the Kalman filter to adapt to the changing behavior of the target.
#
#
# [**Appendix A: Installation, Python, NumPy, and FilterPy**](./Appendix-A-Installation.ipynb)
#
# Brief introduction of Python and how it is used in this book. Description of the companion
# library FilterPy.
#
#
# [**Appendix B: Symbols and Notations**](./Appendix-B-Symbols-and-Notations.ipynb)
#
# Most books opt to use different notations and variable names for identical concepts. This is a large barrier to understanding when you are starting out. I have collected the symbols and notations used in this book, and built tables showing what notation and names are used by the major books in the field.
#
# *Still just a collection of notes at this point.*
#
#
# [**Appendix D: H-Infinity Filters**](./Appendix-D-HInfinity-Filters.ipynb)
#
# Describes the $H_\infty$ filter.
#
# *I have code that implements the filter, but no supporting text yet.*
#
#
# [**Appendix E: Ensemble Kalman Filters**](./Appendix-E-Ensemble-Kalman-Filters.ipynb)
#
# Discusses the ensemble Kalman Filter, which uses a Monte Carlo approach to deal with very large Kalman filter states in nonlinear systems.
#
#
# [**Appendix F: FilterPy Source Code**](./Appendix-F-Filterpy-Code.ipynb)
#
# Listings of important classes from FilterPy that are used in this book.
#
#
# ## Supporting Notebooks
#
# These notebooks are not a primary part of the book, but contain information that might be interested to a subest of readers.
#
#
# [**Computing and plotting PDFs**](./Supporting_Notebooks/Computing_and_plotting_PDFs.ipynb)
#
# Describes how I implemented the plotting of various pdfs in the book.
#
#
# [**Interactions**](./Supporting_Notebooks/Interactions.ipynb)
#
# Interactive simulations of various algorithms. Use sliders to change the output in real time.
#
# [**Converting the Multivariate Equations to the Univariate Case**](./Supporting_Notebooks/Converting-Multivariate-Equations-to-Univariate.ipynb)
#
# Demonstrates that the Multivariate equations are identical to the univariate Kalman filter equations by setting the dimension of all vectors and matrices to one.
#
# [**Iterative Least Squares for Sensor Fusion**](./Supporting_Notebooks/Iterative-Least-Squares-for-Sensor-Fusion.ipynb)
#
# Deep dive into using an iterative least squares technique to solve the nonlinear problem of finding position from multiple GPS pseudorange measurements.
#
#
# [**Taylor Series**](./Supporting_Notebooks/Taylor-Series.ipynb)
#
# A very brief introduction to Taylor series.
#
#
# ### Github repository
# http://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python
#

# In[1]:


# format the book
from book_format import load_style

load_style()
