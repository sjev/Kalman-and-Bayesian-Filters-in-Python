#!/usr/bin/env python
# coding: utf-8

# [Table of Contents](./table_of_contents.ipynb)
\appendix
# # Installation

# This book is written in Jupyter Notebook, a browser based interactive Python environment that mixes Python, text, and math. I choose it because of the interactive features - I found Kalman filtering nearly impossible to learn until I started working in an interactive environment. It is difficult to form an intuition about many of the parameters until you can change them and immediately see the output. An interactive environment also allows you to play 'what if' scenarios. "What if I set $\mathbf{Q}$ to zero?" It is trivial to find out with Jupyter Notebook.
# 
# Another reason I choose it is because most textbooks leaves many things opaque. For example, there might be a beautiful plot next to some pseudocode. That plot was produced by software, but software that is not available to the reader. I want everything that went into producing this book to be available to you. How do you plot a covariance ellipse? You won't know if you read most books. With Jupyter Notebook all you have to do is look at the source code.
# 
# Even if you choose to read the book online you will want Python and the SciPy stack installed so that you can write your own Kalman filters. There are many different ways to install these libraries, and I cannot cover them all, but I will cover a few typical scenarios.

# ## Installing the SciPy Stack

# This book requires IPython, Jupyter, NumPy, SciPy, SymPy, and Matplotlib. The SciPy stack of NumPy, SciPy, and Matplotlib depends on third party Fortran and C code, and is not trivial to install from source code. The SciPy website strongly urges using a pre-built installation, and I concur with this advice.
# 
# Jupyter notebook is the software that allows you to run Python inside of the browser - the book is a collection of Jupyter notebooks. IPython provides the infrastructure for Jupyter and data visualization. NumPy and Scipy are packages which provide the linear algebra implementation that the filters use. Sympy performs symbolic math - I use it to find derivatives of algebraic equations. Finally, Matplotlib provides plotting capability. 
# 
# I use the Anaconda distribution from Continuum Analytics. This is an excellent distribution that combines all of the packages listed above, plus many others. IPython recommends this package to install Ipython. Installation is very straightforward, and it can be done alongside other Python installations you might already have on your machine. It is free to use. You may download it from here: http://continuum.io/downloads I strongly recommend using the latest Python 3 version that they provide, but this book and FilterPy is compatible with Python 3.6 or later.
# 
# There are other choices for installing the SciPy stack. You can find instructions here: http://scipy.org/install.html It can be very cumbersome, and I do not support it or provide any instructions on how to do it.
# 
# Many Linux distributions come with these packages pre-installed. However, they are often somewhat dated and they will need to be updated as the book depends on recent versions of all. Updating a specific Linux installation is beyond the scope of this book. An advantage of the Anaconda distribution is that it does not modify your local Python installation, so you can install it and not break your linux distribution. Some people have been tripped up by this. They install Anaconda, but the installed Python remains the default version and then the book's software doesn't run correctly.
# 
# I do not run regression tests on old versions of these libraries. In fact, I know the code will not run on older versions (say, from 2014-2015). I do not want to spend my life doing tech support for a book, thus I put the burden on you to install a recent version of Python and the SciPy stack. 
# 
# 
# You will need Python 3.6 or later installed.
# 
# Please submit a bug report at the book's [github repository](https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python) if you have installed the latest Anaconda and something does not work - I will continue to ensure the book will run with the latest Anaconda release. I'm rather indifferent if the book will not run on an older installation. I'm sorry, but I just don't have time to provide support for everyone's different setups. Packages like `jupyter notebook` are evolving rapidly, and I cannot keep up with all the changes *and* remain backwards compatible as well. 
# 
# If you need older versions of the software for other projects, note that Anaconda allows you to install multiple versions side-by-side. Documentation for this is here:
# 
# https://conda.io/docs/user-guide/tasks/manage-python.html
# 

# ## Installing FilterPy
# 
# FilterPy is a Python library that implements all of the filters used in this book, and quite a few others. Installation is easy using `pip`. Issue the following from the command prompt:
# 
#      pip install filterpy
#      
#      
# FilterPy is written by me, and the latest development version is always available at https://github.com/rlabbe/filterpy.

# ## Downloading and Running the Book

# The book is stored in a github repository. From the command line type the following:
# 
#     git clone --depth=1 https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python.git
# 
# This will create a directory named Kalman-and-Bayesian-Filters-in-Python. The `depth` parameter just gets you the latest version. Unless you need to see my entire commit history this is a lot faster and saves space.
# 
# If you do not have git installed, browse to https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python where you can download the book via your browser.
# 
# Now, from the command prompt change to the directory that was just created, and then run Jupyter notebook:
# 
#     cd Kalman-and-Bayesian-Filters-in-Python
#     jupyter notebook
# 
# A browser window should launch showing you all of the chapters in the book. Browse to the first chapter by clicking on it, then open the notebook in that subdirectory by clicking on the link.
# 
# More information about running the notebook can be found here:
# 
# http://jupyter-notebook-beginner-guide.readthedocs.org/en/latest/execute.html

# ## Companion Software

# Code that is specific to the book is stored with the book in the subdirectory *./kf_book*. This code is in a state of flux; I do not wish to document it here yet. I do mention in the book when I use code from this directory, so it should not be a mystery.
# 
# In the *kf_book* subdirectory there are Python files with a name like *xxx*_internal.py. I use these to store functions that are useful for a specific chapter. This allows me to hide away Python code that is not particularly interesting to read - I may be generating a plot or chart, and I want you to focus on the contents of the chart, not the mechanics of how I generate that chart with Python. If you are curious as to the mechanics of that, just go and browse the source.
# 
# Some chapters introduce functions that are useful for the rest of the book. Those functions are initially defined within the Notebook itself, but the code is also stored in a Python file that is imported if needed in later chapters. I do document when I do this where the function is first defined, but this is still a work in progress. I try to avoid this because then I always face the issue of code in the directory becoming out of sync with the code in the book. However, IPython Notebook does not give us a way to refer to code cells in other notebooks, so this is the only mechanism I know of to share functionality across notebooks.
# 
# There is an undocumented directory called **experiments**. This is where I write and test code prior to putting it in the book. There is some interesting stuff in there, and feel free to look at it. As the book evolves I plan to create examples and projects, and a lot of this material will end up there. Small experiments will eventually just be deleted. If you are just interested in reading the book you can safely ignore this directory. 
# 
# The subdirectory *./kf_book* contains a css file containing the style guide for the book. The default look and feel of IPython Notebook is rather plain. Work is being done on this. I have followed the examples set by books such as [Probabilistic Programming and Bayesian Methods for Hackers](http://nbviewer.ipython.org/github/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers/blob/master/Chapter1_Introduction/Chapter1.ipynb). I have also been very influenced by Professor Lorena Barba's fantastic work, [available here](https://github.com/barbagroup/CFDPython). I owe all of my look and feel to the work of these projects. 
# 

# ## Using Jupyter Notebook

# A complete tutorial on Jupyter Notebook is beyond the scope of this book. Many are available online. In short, Python code is placed in cells. These are prefaced with text like `In [1]:`, and the code itself is in a boxed area. If you press CTRL-ENTER while focus is inside the box the code will run and the results will be displayed below the box. Like this:

# In[1]:


print(3+7.2)


# If you have this open in Jupyter Notebook now, go ahead and modify that code by changing the expression inside the print statement and pressing CTRL+ENTER. The output should be changed to reflect what you typed in the code cell.

# ## SymPy

# SymPy is a Python package for performing symbolic mathematics. The full scope of its abilities are beyond this book, but it can perform algebra, integrate and differentiate equations, find solutions to differential equations, and much more. For example, we use use it to compute the Jacobian of matrices and expected value integral computations.
# 
# First, a simple example. We will import SymPy, initialize its pretty print functionality (which will print equations using LaTeX). We will then declare a symbol for SymPy to use.

# In[2]:


import sympy
sympy.init_printing(use_latex='mathjax')

phi, x = sympy.symbols('\phi, x')
phi


# Notice how it prints the symbol `phi` using LaTeX. Now let's do some math. What is the derivative of $\sqrt{\phi}$?

# In[3]:


sympy.diff('sqrt(phi)')


# We can factor equations

# In[4]:


sympy.factor(phi**3 -phi**2 + phi - 1)


# and we can expand them.

# In[5]:


((phi+1)*(phi-4)).expand()


# You can evauate an equation for specific values of its variables:

# In[6]:


w =x**2 -3*x +4
print(w.subs(x, 4))
print(w.subs(x, 12))


# You can also use strings for equations that use symbols that you have not defined:

# In[7]:


x = sympy.expand('(t+1)*2')
x


# Now let's use SymPy to compute the Jacobian of a matrix. Given the function
# 
# $$h=\sqrt{(x^2 + z^2)}$$
# 
# find the Jacobian with respect to x, y, and z.

# In[8]:


x, y, z = sympy.symbols('x y z')

H = sympy.Matrix([sympy.sqrt(x**2 + z**2)])

state = sympy.Matrix([x, y, z])
H.jacobian(state)


# Now let's compute the discrete process noise matrix $\mathbf Q$ given the continuous process noise matrix 
# $$\mathbf Q = \Phi_s \begin{bmatrix}0&0&0\\0&0&0\\0&0&1\end{bmatrix}$$
# 
# The integral is 
# 
# $$\mathbf Q = \int_0^{\Delta t} \mathbf F(t)\mathbf Q\mathbf F^T(t)\, dt$$
# 
# where 
# $$\mathbf F(\Delta t) = \begin{bmatrix}1 & \Delta t & {\Delta t}^2/2 \\ 0 & 1 & \Delta t\\ 0& 0& 1\end{bmatrix}$$

# In[9]:


dt = sympy.symbols('\Delta{t}')
F_k = sympy.Matrix([[1, dt, dt**2/2],
                    [0,  1,      dt],
                    [0,  0,       1]])
Q = sympy.Matrix([[0,0,0],
                  [0,0,0],
                  [0,0,1]])

sympy.integrate(F_k*Q*F_k.T,(dt, 0, dt))


# ## Various Links

# https://ipython.org/
# 
# https://jupyter.org/
# 
# https://www.scipy.org/
