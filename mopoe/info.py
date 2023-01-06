# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2021
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

# Module current version
version_major = 0
version_minor = 0
version_micro = 1

# Expected by setup.py: string of form "X.Y.Z"
__version__ = "{0}.{1}.{2}".format(version_major, version_minor, version_micro)

# Expected by setup.py: the status of the project
CLASSIFIERS = ["Development Status :: 5 - Production/Stable",
               "Environment :: Console",
               "Environment :: X11 Applications :: Qt",
               "Operating System :: OS Independent",
               "Programming Language :: Python",
               "Topic :: Scientific/Engineering",
               "Topic :: Utilities"]

# Project descriptions
description = """
Generalized Multimodal ELBO.
"""
SUMMARY = """
.. container:: summary-carousel

    Generalized Multimodal ELBO.
"""
long_description = (
    "Generalized Multimodal ELBO.\n")

# Main setup parameters
NAME = "mopoe"
ORGANISATION = "CEA"
MAINTAINER = "Antoine Grigis"
MAINTAINER_EMAIL = "antoine.grigis@cea.fr"
DESCRIPTION = description
LONG_DESCRIPTION = long_description
EXTRANAME = "NeuroSpin webPage"
EXTRAURL = (
    "https://joliot.cea.fr/drf/joliot/Pages/Entites_de_recherche/"
    "NeuroSpin.aspx")
LINKS = {"scripts": "https://github.com/neurospin/scripts"}
URL = "https://github.com/neurospin-deepinsight/mopoe"
DOWNLOAD_URL = "https://github.com/neurospin-deepinsight/mopoe"
LICENSE = "MIT"
AUTHOR = """
mopoe developers
"""
AUTHOR_EMAIL = "antoine.grigis@cea.fr"
PLATFORMS = "OS Independent"
ISRELEASE = True
VERSION = __version__
PROVIDES = ["mopoe"]
REQUIRES = [
    "matplotlib",
    "numpy",
    "pandas",
    "torch",
    "scikit-learn",
    "scipy",
    "tensorboardx",
    "torchvision",
    "fire",
    "iterative-stratification"
]
SCRIPTS = [
    "mopoe/mopoe"
]
