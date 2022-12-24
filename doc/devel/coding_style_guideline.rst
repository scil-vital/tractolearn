==================================
tractolearn Coding Style Guideline
==================================

The main principles behind tractolearn development are:

* **Robustness**: the results of a piece of code must be verified
  systematically, and hence stability and robustness of the code must be
  ensured, reducing code redundancies.
* **Readability**: the code is written and read by humans, and it is read
  much more frequently than it is written.
* **Consistency**: following these guidelines will ease reading the code,
  and will make it less error-prone.
* **Documentation**: document the code. Documentation is essential as it helps
  clarifying certain choices, helps avoiding obscure places, and is a way to
  allow other members *decode* it with less effort.
* **Language**: the code must be written in English. Norms and spelling
  should be abided by.

Largely inspired from `DIPY <https://dipy.org/>`_.

Coding style
============

tractolearn uses the standard Python `PEP8`_ style to ensure the readability and
consistency across the toolkit. Conformance to the PEP8 syntax is checked
automatically when requesting to push to tractolearn. There are
`software tools <https://pypi.python.org/pypi/pep8>`_ that check your code for
PEP8 compliance, and most text editors can be configured to check the
compliance of your code with PEP8.

In tractolearn The code style is automatically enforced using the `black`_
package.

Imports
-------

tractolearn recommends using the following package aliases to increase
consistency and readability across the library::

    import matplotlib.pyplot as plt
    import nibabel as nib
    import numpy as np
    import numpy.testing as npt

No alias should be used for ``h5py``::

    import h5py

Stick to the *common* or *recommended* import practices for each standard
packages::

    from datetime import (datetime, timedelta)

It is understood as *common* or *recommended* practice one that is used in the
documentation or examples of the package.

Use generic imports for other packages::

    import package
    import package.subpackage.subsubpackage

In tractolearn, imports are automatically sorted using the `isort`_ package.

Naming guidelines
-----------------

The following is a set of conventions to make the tractolearn to improve the
code readability and make it consistent.

General considerations
^^^^^^^^^^^^^^^^^^^^^^

* Keep name length within a four word maximum, and avoid gratuitous context.
* Only use correctly-spelled dictionary words and abbreviations that appear in a
  dictionary. Make exceptions for ``id`` and documented domain-specific
  abbreviations (e.g. ``fa``, ``md``, ``acc``, ``pos``).
* Names are usually read left-to-right, from more specific to less specific:
  e.g. ``L2RegularizedSphericalDeconvolutionFilter`` instead of
  ``SphericalDeconvolutionL2RegularizedFilter``.

Package names
^^^^^^^^^^^^^

* Use a noun-action name: e.g. ``learning`` instead of ``learners``. An
  exception could be a package named ``models``.

Class names
^^^^^^^^^^^

* Use a noun-phrase name: e.g. ``GaussianNoiseRemovalFilter`` instead of
  ``GaussianNoiseFiltering``.

Method names
^^^^^^^^^^^^

* Use a verb-phrase for method names: e.g. ``flip_streamlines`` (instead of
  ``streamline_flip``).
* Use the ``get`` prefix for field accessors that return a value.
* Use ``is`` and ``has`` prefixes for boolean field accessors.
* Use the ``set`` prefix for field accessors that do not return a value.
* Use validation verbs for methods that provide the result: use verbs like
  ``validate``, ``check`` or ``ensure`` to name methods that either result or
  throw an exception when validation fails.
* Use transformation verbs for methods that return a transformed value: use
  verbs that suggest transformation, like ``convert``, for methods that return
  the result.
* Use the ``compute`` prefix for methods that compute and return a value.
* Use opposites precisely, e.g. ``add/remove``, ``begin/end``, ``first/last``,
  ``insert/delete``, etc.
* Use the name ``create`` instead of ``generate`` for methods that *create*
  something.

Variables
^^^^^^^^^

* Qualification: qualify values with suffixes. Use a suffix to describe what
  kind of value constant and variable values represent. Suffixes such as
  ``min`` (for maximum) or ``min`` (for minimum), ``count`` (instead of ``num``
  or ``nb``) and ``avg`` (for average) relate a collection of values to a single
  derived value. Using a suffix, rather than a prefix, for the qualifier
  naturally links the name to other similar names: e.g.
  ``streamline_lengths_max`` instead of ``max_streamline_lengths``, or
  ``streamline_count`` instead of ``count_streamlines`` (which would be a method
  name).
* Avoid adding the type of a variable to its name, i.e. do not use::

    streamlines_list = []

  Instead, use::

    streamlines = []

* Limit the use of plural words to collections, but prefer collective nouns for
  collections whenever possible. If a variable comprising two names needs to b
  pluralized, pluralize the last term, i.e. ``streamline_lengths`` instead of
  ``streamlines_length``.
* Use boolean variable names that imply true or false: use names like ``done``
  or ``found`` that describe boolean values.
* Replace boolean names with names in the correct grammatical form: do not use
  negation in boolean names. Do not use names that require a prefix like ``not``
  that inverts the variable's truth value.

Documentation
-------------

tractolearn uses `Sphinx`_ to generate documentation.

tractolearn follows the `NumPy docstring standard`_ for documenting modules,
classes, functions, and examples.

Particularly, with the consistency criterion in mind, beyond the `NumPy docstring standard`_
aspects, contributors are encouraged to observe the following guidelines:

* The classes, objects, and any other construct referenced from the code
  should be written with inverted commas.
* Use an all-caps scheme for acronyms, and capitalize the first letters of
  the long names, such as in *Constrained Spherical Deconvolution (CSD)*,
  except in those cases where the most common convention has been to use
  lowercase, such as in *superior longitudinal fascicle (SLF)*.
* As customary in Python, use lowercase and separate words with underscores
  for filenames, labels for references, etc.

For now, docstrings are not being checked by the automatic code style tools.


Enforcing the coding style
==========================

tractolearn uses `pre-commit`_ to ensure that the code complies with those
aspects that can be automatically checked and enforced. `pre-commit`_ is a
Python package that when installed and configured runs the pre-commit hooks
specified through a configuration file named ``.pre-commit-config.yaml``.

.. Links
.. Python-related tools
.. _black: https://black.readthedocs.io/en/stable/?badge=stable
.. _isort: https://pycqa.github.io/isort/
.. _`NumPy docstring standard`: https://numpydoc.readthedocs.io/en/latest/format.html
.. _`pre-commit`: https://pre-commit.com/
.. _PEP8: https://www.python.org/dev/peps/pep-0008/
.. _Sphinx: http://www.sphinx-doc.org/en/stable/index.html
