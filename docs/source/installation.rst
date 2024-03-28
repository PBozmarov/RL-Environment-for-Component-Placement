Installation
===========

This repository runs on Python 3.9. Due to compatibility issues with Tensorflow and Windows, the repository is only supported on Linux and MacOS. Additionally, due to compatibility issues with Tensorflow and MacOS separate requirements files are provided for Linux and MacOS.

.. _linux:

Linux Installation
----------------------------

To install and use the repository on Linux, run the following shell commands:

.. code-block:: bash

   git clone https://github.com/kiaashour/InstaDeep-Software-Engineering-Project.git
   cd InstaDeep-Software-Engineering-Project
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements/requirements-linux.txt


.. _macos:

MacOS Installation
----------------------------

Download Miniconda
~~~~~~~~
Download Miniconda from the following URL: https://docs.conda.io/en/latest/miniconda.html_

.. note:: If this step doesn't work, you can try to skip it, but it is not guaranteed that the below steps will work.

Miniconda
~~~~~~~~
Next, you should install the xcode-select command-line utilities. Use the following command to install:

.. code-block:: bash

   xcode-select --install

.. warning:: If the above command gives an error, you should install XCode from the App Store. You can skip this step (the Jupyter one) if you have it.


Deactivate the Base Environment
~~~~~~~~
First, we need to deactivate the base environment.

.. code-block:: bash

   conda deactivate

Create the New Environment
~~~~~~~~
Next, we will install the Apple Silicon tensorflow.yml file provided. Run the following command from the same directory that contains tensorflow.yml. 

.. code-block:: bash

   cd requirements
   conda env create -f tensorflow.yml

Activate the New Environment
~~~~~~~~
To enter this environment, you must use the following command:

.. code-block:: bash

   conda activate tensorflow-apple

Install Dependencies from Requirements
~~~~~~~~
Now, install the dependencies from the requirements-macos.txt file.

.. code-block:: bash

   pip install -r requirements-macos.txt


.. note:: A separate requirements file is provided for development, ```requirements/requirements-dev.txt```. 