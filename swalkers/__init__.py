'''
swalkers package initialization

The 's' in 'swalkers' stands both for 'smart' and 'simple', since this code is made to work only in 1D.
Even if the techniques reported are good for any dimension.
'''

__version__ = "0.1.0"
__author__ = "Gianluca Peri"

# Import all modules in the swalkers package

from .entities import *
from .functions import *
from .simulation import *
from .utilities import *