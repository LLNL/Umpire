.. ##############################################################################
.. # Copyright (c) 2016-25, Lawrence Livermore National Security, LLC and Umpire
.. # project contributors. See the COPYRIGHT file for details.
.. #
.. # SPDX-License-Identifier: (MIT)
.. ##############################################################################

.. _governance-policy:

=================
Governance Policy
=================

Introduction
============

Umpire is an application-focused API for memory management on NUMA & GPU architectures. This governance policy outlines the rules and processes that guide the development, contributions, and decision making of the Umpire project.

Roles and Responsibilities
==========================

Project Maintainers
~~~~~~~~~~~~~~~~~~~

- **Definition**: A group of trusted contributors responsible for the overall direction and health of the project and who may be members of the Technical Steering Committee (see below).
- **Responsibilities**:

  - Reviewing and merging pull requests.
  - Ensuring the project adheres to its code of conduct.
  - Managing releases and ensuring high-quality standards.
  - Facilitating discussions and resolving project issues.

Contributors
~~~~~~~~~~~~

- **Definition**: Individuals who contribute to the project, including code, documentation, and other assets.
- **Members**: See the list of contributors for Umpire `here <https://github.com/LLNL/Umpire/graphs/contributors>`_.
- **Responsibilities**:

  - Following the contribution guidelines.
  - Participating in discussions and code reviews.
  - Reporting issues and suggesting improvements.

Technical Steering Committee
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Definition**: A subset of maintainers responsible for making high-level decisions.
- **Members**: David Beckingsale and Kristi Belcher
- **Responsibilities**:

  - Setting the long-term technical and community goals of the project.
  - Deciding on major feature inclusions and deprecations.
  - Resolving escalated bug fix or feature request disputes.

Contribution Process
====================

Pull Requests
~~~~~~~~~~~~~

- Contributions are made via pull requests (PRs).
- PRs must be reviewed by at least one maintainer before merging.
- See the Contribution guide `here <https://github.com/LLNL/Umpire/blob/develop/CONTRIBUTING.md>`_.

Issues
~~~~~~

- Contributors are encouraged to create issues for bugs, feature requests, and questions.
- Maintainers triage issues regularly and assign priorities.

Code of Conduct
~~~~~~~~~~~~~~~

- All contributors must adhere to the project's `Code of Conduct <https://github.com/LLNL/Umpire/blob/develop/CODE_OF_CONDUCT.md>`_.

Release Management
==================

Regular Releases
~~~~~~~~~~~~~~~~

- Umpire follows a regular release schedule (see below).
- Each release includes a summary of changes, new features, and bug fixes.
- Umpire maintainers will coordinate project releases according to the release schedule. 
- Releases are coordinated with the RAJA and Camp teams as part of timely RAJA Portability Suite releases. 
- The release names will correspond to the release names of RAJA and Camp as part of this process. Once the release has been merged, it will be published in the `Releases <https://github.com/LLNL/Umpire/releases>`_ section of the repo. 

Communication Channels
======================

- The primary communication channel is the GitHub repository (issues, PR comments).
- Other channels may include mailing lists and Slack (See the `README <https://github.com/LLNL/Umpire/blob/develop/README.md>`_ for details).

Amendments
==========

- Changes to this governance policy require a formal proposal and approval by the Technical Steering Committee.

