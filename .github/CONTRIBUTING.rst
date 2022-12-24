===========================
Contributing to tractolearn
===========================

Welcome to tractolearn !

This article documents how to contribute improvements to tractolearn.

The development process
=======================

When writing code, please pay attention to the following aspects:

Git Hooks
---------

A number of git hooks have been developed to fire some actions when certain git
actions occur. In order to set up these hooks, it is necessary to run::

    tools/gitsetup/setup-hooks.sh


Requirements
------------

In order to keep the package healthy, a number of additional packages need to
be installed so that the code format and tests are checked.

Such packages can be installed running::

   pip install pre-commit


Testing
-------

`pytest`_ is used to write the tests of the code.

If you are adding code into a module that already has a test file, add
additional tests into the respective file.

Coverage
--------

New contributions are required to have as close to 100% code coverage as
possible. This means that the tests written cause each and every statement in
the code to be executed, covering corner-cases, error-handling, and logical
branch points.

When running::

    coverage run -m pytest -s --doctest-modules --verbose tractolearn

You will get the usual output of `pytest`_, but also a table that indicates the
test coverage in each module: the percentage of coverage and also the lines of
code that are not run in the tests. You can also see the test coverage in the
corresponding CI build of the pull request (PR).

If your contributions are to a single module, you can see test and
coverage results for only that module without running all of the tractolearn
tests. For example, if you are adding code to ``tractolearn/core/utils.py``,
you can use::

    coverage run --source=tractolearn.core.utils -m pytest -s --doctest-modules --verbose tractolearn/core/tests/test_utils.py

You can then use ``coverage report`` to view the results, or use
``coverage html`` and open ``htmlcov/index.html`` in your browser for a nicely
formatted interactive coverage report.

Code style
----------

Code contributions should be formatted according to the `tractolearn Coding Style Guideline <../doc/devel/coding_style_guideline.rst>`_.
Please, read the document to conform your code contributions to the tractolearn
standard.s

Documentation
-------------

tractolearn uses `Sphinx`_ to generate documentation. The `tractolearn Coding Style Guideline <../doc/devel/coding_style_guideline.rst>`_
contains details about documenting the contributions.

To build the documentation locally, you can build it running::

    make html

Unless otherwise stated the documentation is built under ``build/html/``. You
can open the ``index.html`` file to view the documentation generated from your
local copy.

Commit Messages
===============

Write your commit messages using the standard prefixes for tractolearn commit
messages:

* ``BUG:`` Fix for runtime crash or incorrect result
* ``COMP:`` Compiler error or warning fix
* ``DOC:`` Documentation change
* ``ENH:`` New functionality
* ``PERF:`` Performance improvement
* ``STYLE:`` No logic impact (indentation, comments)
* ``WIP:`` Work In Progress not ready for merge

The body of the message should clearly describe the motivation of the commit
(**what**, **why**, and **how**). In order to ease the task of reviewing
commits, the message body should follow the following guidelines:

1. Leave a blank line between the subject and the body. This helps ``git log``
   and ``git rebase`` work nicely, and allows to smooth generation of release
   notes.
2. Try to keep the subject line below 72 characters, ideally 50.
3. Capitalize the subject line.
4. Do not end the subject line with a period.
5. Use the imperative mood in the subject line (e.g. ``BUG: Fix spacing not
   being considered.``).
6. Wrap the body at 80 characters.
7. Use semantic line feeds to separate different ideas, which improves the
   readability.
8. Be concise, but honor the change: if significant alternative solutions were
   available, explain why they were discarded.
9. If the commit fixes a regression test, provide the link. If it fixes a
   compiler error, provide a minimal verbatim message of the compiler error. If
   the commit closes an issue, use the `GitHub issue closing keywords <https://help.github.com/en/articles/closing-issues-using-keywords>`_.

Keep in mind that the significant time is invested in reviewing commits and
pull requests, so following these guidelines will greatly help the people doing
reviews.

These guidelines are largely inspired by Chris Beam's `How to Write a Commit Message <https://chris.beams.io/posts/git-commit/>`_
post, and `ITK <https://itk.org/>`_ (including self contributions to ITK).


.. Links
.. Python-related tools
.. _pytest: https://docs.pytest.org
.. _Sphinx: http://www.sphinx-doc.org/en/stable/index.html
