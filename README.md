# Sales price prediction

Use regression to predict price of electronic devices

Tip: If you don't have markdown viewer like Atom, you can render this on Chrome by following [this link](https://markdownlivepreview.com/).

# Setup environment using task utility (Recommended, with UV)
## Pre-requisites

To develop Sales price prediction, the following tools must be installed on your system:

- `git`: required for version control
- `task`: required for using the automation setup
- `direnv`: required for handling environment variables
- `uv`: required for managing python environments

**Linux** is the preferred platform for building Sales price prediction, however the instructions are tested to work on **Windows** as well. If you're someone having access to a Windows machine only, try to setup **Ubuntu** using **Windows Sub-System for Linux (WSL)**.

The step-by-step platform specific installation instructions of the mentioned tools are presented here:

## Linux (WSL/Ubuntu)

### 1. Updating package cache

Before installing any tools on Linux, it is essential to update the package cache. This ensures that your system has the latest information about available softwares and versions from the repositories. Run the following command in the terminal to update the package cache:

```bash
sudo apt update -y
```

### 2. Installing git

To install `git`, run the following command:

```bash
sudo apt install git -y
```

You can verify the installation by asking for the `git` version with `git --version`.

### 3. Installing task

To install `task`, run the following commands:

```bash
# downloads and installs the task utility to the /home/<user>/bin directory
sh -c "$(curl --location https://taskfile.dev/install.sh)" -- -d -b $HOME/bin

# writes the path to the /home/<user>/.bashrc file and sources it
echo 'export PATH="$PATH:$HOME/bin"' >> $HOME/.bashrc && source $HOME/.bashrc
```
As before, you can verify the installation by asking for the `task` version with `task --version`.

### 4. Installing direnv

To install `direnv`, run the following commands:

```bash
# downloads and installs direnv
sudo apt install direnv -y

# configures direnv
echo 'eval "$(direnv hook bash)"' >> ~/.bashrc && source ~/.bashrc
```

As before, you can verify the installation by asking for the `direnv` version with `direnv --version`.

### 5. Installing uv

To install `uv`, run the following commands:

```bash
# downloads and installs uv
curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$HOME/.local/bin" sh

# writes the path to the /home/<user>/.bashrc file and sources it
echo 'export PATH="$HOME/.local/bin:$PATH"' >> $HOME/.bashrc && source $HOME/.bashrc
```

As usual, you can verify the installation by asking for the `uv` version with `uv --version`.

### 6. Installing tools required for pyspark

> **Note:** Installing `JDK` and `Spark` is optional unless you plan to develop with Spark..

`Sales price prediction` supports working with Spark. To setup `spark` properly, follow the instructions given below:

**Installing and configuring the Java Development Kit (JDK)**

This section covers the installation and configuration of the Java Development Kit(JDK) on the system. As of now, `openjdk-11-jdk` is tested and used.

```bash
# downloads and installs the open java development kit
sudo apt install openjdk-11-jdk -y

# verify installation (optional)
java --version && javac --version

# check java path (optional)
update-alternatives --config java

# define the JAVA_HOME environment variable and write to `$HOME/.bashrc`
echo 'export "JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64"' >> $HOME/.bashrc

# extend PATH variable by appending JAVA_HOME
echo 'export PATH="$PATH:$JAVA_HOME/bin"' >> $HOME/.bashrc

# reload $HOME/.bashrc
source $HOME/.bashrc

# verify JAVA_HOME value (optional)
echo $JAVA_HOME
```

**Installing and configuring the Spark binary**

This part covers the installation and configuration of spark binary on the system.

As of now, `spark-3.5.6` is getting used, however the version might differ. Please visit the [Spark Download](https://spark.apache.org/downloads.html) page for more information.

```bash
# download pre-built spark binary
wget https://dlcdn.apache.org/spark/spark-3.5.6/spark-3.5.6-bin-hadoop3.tgz

# extract spark binary
tar -xvzf spark-3.5.6-bin-hadoop3.tgz

# move spark binary to /opt/spark
sudo mv spark-3.5.6-bin-hadoop3 /opt/spark

# define the SPARK_HOME variable and write to $HOME/.bashrc
echo 'export SPARK_HOME=/opt/spark' >> $HOME/.bashrc

# extend PATH variable by appending SPARK_HOME/bin and SPARK_HOME/sbin
echo 'export PATH=$SPARK_HOME/bin:$SPARK_HOME/sbin:$PATH' >> $HOME/.bashrc

# reload ~/.bashrc
source $HOME/.bashrc

# verify SPARK_HOME value (optional)
echo $SPARK_HOME
```

## Windows

### 1. Installing git

Download [Git for Windows](https://gitforwindows.org/) and install it as an non-admin user. You may need to set the git installation path as an user environment variable before you start using it.

Installing git on Windows also brings `Git Bash` which is a lightweight terminal emulator that provides a **Unix-like shell environment on Windows**.

Therefore `Git Bash` must be used to run the tasks on the Windows platform.

### 2. Installing task

Download [Task for Windows](https://github.com/go-task/task/releases) i.e. `task_windows_amd64.zip`, then extract and keep the `task.exe` inside `C:\Users\<User.Name>\tools` folder.

### 3. Installing direnv

Download [Direnv for Windows](https://github.com/direnv/direnv/releases) i.e. `direnv.windows-amd64.exe`, then rename it as `direnv.exe` and place it inside `C:\Users\<User.Name>\tools` folder.

To configure `direnv` with `Git Bash`, create a text file as follows `C:\Users\<User.Name>\.bashrc` and add the content shown below:

```bash
# write inside the C:\Users\<User.Name>\.bashrc file
eval "$(direnv hook bash)"
```

To verify the configuration, use `direnv status` on `Git Bash` shell.

### 4. Installing uv

Download [UV for Windows](https://github.com/astral-sh/uv/releases) i.e. `uv-x86_64-pc-windows-msvc.zip`, then extract the `uv.exe`, `uvw.exe` and `uvx.exe` files and place them inside `C:\Users\<User.Name>\tools` folder.

### 5. Setting the Installation Path Using Git Bash Shell

To make all the executables available for use, the path `C:\Users\<User.Name>\tools` has to be made available to `Git Bash` which can be done as shown below:

```bash
# write inside the C:\Users\<User.Name>\.bashrc file
export PATH="$PATH:/c/Users/<User.Name>/tools/"
```

And then reload the shell with `source ~/.bashrc`.

To verify the installations for all the tools, run the commands as shown below:

```bash
# use git bash shell

# verify git installation
git --version

# verify task installation
task --version

# verify direnv installation
direnv --version && direnv status

# verify uv installation
uv --version
```

  > **Note**
  >
  > Windows firewall configured on Tiger assets restricts downloading of python packages and to avoid that UV uses `--native-tls` option for both `dev.setup-env` and `dev.setup-env-pyspark` for the Windows platform. This feature drops UV's performance on Windows.
  >

## Getting started

* Switch to the root folder (i.e. folder containing this file)
* Inside `Sales price prediction`, we use the [task utility](https://taskfile.dev) to manage various workflow automation tasks with a common and platform agnostic interface. It acts as a wrapper for various commands for both Linux and Windows.

* The task utility relies on the task configurations available in `Taskfile.yml` available inside the project root.

```
~/<proj-folder>$ task -l
```

* To verify pre-requisites, run

```
~/<proj-folder>$ task debug.check-reqs
```

### Help Command

- The `task` command has a built-in help facility available for each of the task builtins. To use it, type `help=true` followed by the command:

    ```
    ~/<proj-folder>$ task <command> help=true
    ```
- On running the ``help=true`` command, you get to see the different options supported by it.

## Environment setup:

### Introduction
* Environment is divided into three sections

    * Core - These are must-have packages and will be set up by default. They are listed under the `dependencies` section in `pyproject.toml` and will be installed by default.
    * Usecases - These are for specific usecases you can choose to install. Here are the usecase options.
        * `ebo` – To support Emerging Business Opportunity workflows in Python.
        * `mmx` – To enable Market Mix Modeling capabilities using Python.
        * `pyspark` – To support classification and regression models using PySpark.
        * `reco` – To build Recommendation Systems using Python.
        * `rtm` – To enable Route To Market analysis and modeling in Python.
        * `tpo` – To perform Trade Promotion Optimization using Python.
    * Addons - These are for specific purposes you can choose to install. Here are the addon options
        * `formatting` - To enforce coding standards in your projects.
        * `documentation` - To auto-generate doc from doc strings and/or create rst style documentation to share documentation online
        * `testing` - To use automated test cases.
        * `build` - To build the Python package into distribution archives (e.g., .whl and .tar.gz).
        * `jupyter` - To run the notebooks. This includes jupyter extensions for spell check, advances formatting.
        * `extras` - To include optional utilities for extended functionality.
        * `ts_dev` - Install this to work with time series data.
        * `tareg_dev` – To support development for the Tareg module.
        * `bayesian_dev` – To support Bayesian modeling and probabilistic programming.

    * Edit the dependencies in `pyproject.toml` as needed.
    * The usecases `ebo`, `mmx`, `pyspark`, `reco`, `rtm`, and `rpo` are listed under the `[project.optional-dependencies]`
    * The addons `formatting`, `documentation`, `testing`, and `build` groups are defined under the `[dependency-groups]` section.
    * The addons `bayesian_dev`, `jupyter`, `extras`, `ts_dev`, and `tareg_dev` are listed under the `[project.optional-dependencies]` section.
* You can edit them to your need. All these packages including addons are curated with versions & tested thoroughly for acceleration.
* While you can choose, please decide upfront for your project and everyone use the same options.
* Below you can see how to install the core environment & addons separately. However, we strongly recommend to update the core env with the addons packages as needed for your project. This ensures there is only one version of the env file for your project.
* **To run the reference notebooks and production codes, it is recommended to install all addons.**

## Setup a development environment:

* Run below to install core libraries
    ```
    ~/<proj-folder>$ task dev.setup-env usecase=<specific usecase>
    ```

    The above command should create a uv python environment named `ta-lib-dev` and install the code in the current repository along with all required dependencies.

    You can customize the environment name by replacing the default ta-lib-dev with a name of your choice.
    Make sure to use the -dev suffix so that the environment name ends with dev.
    Example: my-env-dev

    ```
    ~/<proj-folder>$ task dev.setup-env env_name=my-env-dev  usecase=<specific usecase>
    ```
* `usecase` parameter above is an optional parameter. It takes a value of `tpo` or `mmx` or `ebo` or `rtm` or `reco`. `dev.setup-env` in itself will only install the core libs required but when you have to work with specific usecase (e.g MMX or TPO, etc.), one has to install the libraries required for these specific usecases. So when we provide the `usecase` option, we are specifying that we
want that dependencies for this usecase installed in our environment as well.

* To create a `uv` environment for the `pyspark` usecase, use the command below.
  In the `pyspark` setup, you can also specify additional arguments like `usecase` and `addon`, as shown below.
    ```
    ~/<proj-folder>$ task dev.setup-env-pyspark usecase=<specific usecase>
    ```
     You can customize the environment name by replacing the default ta-lib-pyspark-dev with a name of your choice.
    Make sure to use the -dev suffix so that the environment name ends with dev.
    Example: my-env-pyspark-dev

    ```
    ~/<proj-folder>$ task dev.setup-env-pyspark env_name=my-env-pyspark-dev  usecase=<specific usecase>
    ```

## Creating UV Environment Without Specifying a Usecase:

In this case the core packages will be installed from `dependencies` in `pyproject.toml`.
The below command will create a uv environment with python version `3.12`.

```
~/<proj-folder>$ task dev.setup-env python=3.12
```
`python` parameter above is an optional parameter. By default it is set to `3.12` but it can take values of `3.10`, `3.11` or `3.12`.

## Creating UV Environment Specifying a Usecase:
* The following command will create a `uv` environment for a specific usecase. For the `pyspark` usecase, use the `dev.setup-env-pyspark` task to create a PySpark-specific environment.

    ```
    ~/<proj-folder>$ task dev.setup-env usecase=<usecase>
    ```

* Activate the environment first to install other addons. Keep the environment active for all the remaining commands in the manual.

    ```
    ~/<proj-folder>$ source ta-lib-dev/bin/activate
    ```

* Now run all following command to install all the addons. Feel free to customize addons as suggested in the introduction.

    ```
    (ta-lib-dev):~/<proj-folder>$ task dev.setup-addon formatting=true build=true documentation=true testing=true
    ```

* Using `task dev.setup-addon` utility you can install other addons like `bayesian_dev`, `extras`, `jupyter`, `tareg_dev`, or `ts`.
    ```
    (ta-lib-dev):~/<proj-folder>$ task dev.setup-addon env_name=ta-lib-dev addon=<addon-name>
    ```

You now should have a standalone uv python environment and installed code in the current repository along with all required dependencies.

* Get the installation info by running
    ```
    (ta-lib-dev):~/<proj-folder>$ task dev.info
    ```

* Test your installation by running
    ```
    (ta-lib-dev):~/<proj-folder>$ task test.val-env usecase=<specific usecase>
    ```

We need to specify the usecase to validate the environment for core as well as usecase specific dependencies.

* This will just check the core setup, i.e, the env setup by task dev.setup-env
* To check the addon installation in the uv env, we check it by specifying the specific addon like

    ```
    (ta-lib-dev):~/<proj-folder>$ task test.val-env formatting=true documentation=true testing=true build=true
    ```

## Launching Jupyter Notebooks

- In order to launch a jupyter notebook locally in the web server, run

    ```
    (ta-lib-dev):~/<proj-folder>$ task launch.jupyterlab
    ```
     After running the command, type [localhost:8080](localhost:8080) to see the launched JupyterLab.

> **Note:**
> * To install the `glmnet` package, use a Conda environment, as it is not compatible with `uv`.
> * To use the **ebo** use case, install `patchwork` along with `invoke<2.0.0` and `fabric==2.7.3`.
> * The classification PySpark template notebooks have been updated with optimized Spark configurations (`spark.driver.memory` and `spark.sql.autoBroadcastJoinThreshold`) in `notebooks/reference/conf/config.yml`. These settings are pre-configured to improve stability and performance but can be adjusted based on your workload requirements.


# Setup environment using invoke utility (for Conda environments)
## Pre-requisites

* Ensure you have `Miniconda` installed and can be run from your shell. If not, download the installer for your platform here: https://docs.conda.io/en/latest/miniconda.html

     **NOTE**

     * If you already have `Anaconda` installed, go ahead with the further steps, no need to install miniconda.
     * If `conda` cmd is not in your path, you can configure your shell by running `conda init`.


* Ensure you have `git` installed and can be run from your shell

     **NOTE**

     * If you have installed `Git Bash` or `Git Desktop` then the `git` cli is not accessible by default from cmdline.
       If so, you can add the path to `git.exe` to your system path. Here are the paths on a recent setup

        ```
        %LOCALAPPDATA%\Programs\Git\git-bash.exe
        %LOCALAPPDATA%\GitHubDesktop\app-<ver>\resources\app\git\mingw64\bin\git.exe
        ```

* Ensure [invoke](http://www.pyinvoke.org/index.html) tool and pyyaml are installed in your `base` `conda` environment. If not, run

    ```
    (base):~$ pip install invoke
    (base):~$ pip install pyyaml
    ```
* Use **invoke** version `2.2.0`. To verify the installed version:
  `invoke --version`
* If the version does not match, install the correct version along with the package in editable mode:
  `pip install invoke==2.2.0`

## Getting started

* Switch to the root folder (i.e. folder containing this file)
* A collection of workflow automation tasks can be seen as follows
    **NOTE**
     * Please make sure there are no spaces in the folder path. Environment setup fails if spaces are present.

    ```
    (base):~/<proj-folder>$ inv -l
    ```

* To verify pre-requisites, run

    ```
    (base)~/<proj-folder>$ inv debug.check-reqs
    ```

and check no error messages (`Error: ...`) are printed.

### Help Command

- The `inv` command has a built-in help facility available for each of the invoke builtins. To use it, type `--help` followed by the command:

    ```
    (ta-lib-dev):~/<proj-folder>$ inv <command> --help
    ```
- On running the ``help`` command, you get to see the different options supported by it.

## Environment setup:

### Introduction
* Environment is divided into three sections

    * Core - These are must-have packages and will be set up by default. They are declared in `deploy/pip/ct-core-dev.txt` and the same packages are also declared in `deploy/conda_envs/ct-core-dev.yml`.
    * Usecases - These are for specific usecases you can choose to install. Here are the usecase options.
        * `ebo` – To support Emerging Business Opportunity workflows in Python.
        * `mmx` – To enable Market Mix Modeling capabilities using Python.
        * `pyspark` – To support classification and regression models using PySpark.
        * `reco` – To build Recommendation Systems using Python.
        * `rtm` – To enable Route To Market analysis and modeling in Python.
        * `tpo` – To perform Trade Promotion Optimization using Python.
    * Addons - These are for specific purposes you can choose to install. Here are the addon options
        * `formatting` - To enforce coding standards in your projects.
        * `documentation` - To auto-generate doc from doc strings and/or create rst style documentation to share documentation online
        * `testing` - To use automated test cases
        * `jupyter` - To run the notebooks. This includes jupyter extensions for spell check, advances formatting.
        * `extras` - there are nice to haves or for pointed usage.
        * `ts` - Install this to work with time series data
        * `ts_dev` - Install this to work with time series data.
        * `tareg_dev` – To support development for the Tareg module.
        * `bayesian_dev` – To support Bayesian modeling and probabilistic programming.
    * Edit the addons in `deploy/pip/addon-<addon-name>-dev.txt` and the usecases in `deploy/pip/ct-<usecase-name>-dev.txt` to suit your need.
    * Each of the packages there have line comments with their purpose. From an installation standpoint extras are treated as addons
* You can edit them to your need. All these packages including addons & extras are curated with versions & tested thoroughly for acceleration.
* While you can choose, please decide upfront for your project and everyone use the same options.
* Below you can see how to install the core environment & addons separately. However, we strongly recommend to update the core env with the addons packages & extras as needed for your project. This ensures there is only one version of the env file for your project.
* **To run the reference notebooks and production codes, it is recommended to install all addons.**
* Tip: Default name of the env is `ta-lib-dev`. You can change it for your project.
    * For example: to make it as `env-myproject-prod`.
    * Open `tasks.py`
    * Set `ENV_PREFIX = 'env-customer-x'`
    * `dev` suffix will be added automatically after environment creation.

## Setup a development environment

Run below to install core libraries
```
(base):~/<proj-folder>$ inv dev.setup-env --usecase=<specific usecase>
```

The above command should create a conda python environment named `ta-lib-dev` and install the code in the current repository along with all required dependencies.

`usecase` parameter above is an optional parameter. It takes a value of `tpo` or `mmx` or `ebo` or `rtm` or `reco`.
dev.setup-env in itself will only install the core libs required but when you have to work
with specific usecase (e.g MMX or TPO, etc.), one has to install the libraries required for
these specific usecases. So when we provide the `usecase` option, we are specifying that we
want that dependencies for this usecase installed in our environment as well.

To create a `conda` environment for the `pyspark` usecase, use the command below.
  In the `conda` setup, you can also specify additional arguments like `usecase` and `addon`, as shown below.
    ```
    ~/<proj-folder>$ inv dev.setup-env-pyspark --usecase=<usecase>
    ```
* Activate the respective PySpark environment (virtualenv, conda, or other).
* Verify whether PySpark is installed:
   `pip show pyspark`
* If PySpark is not installed, install it along with development dependencies in editable mode:

  ```bash
  pip install -e . -r deploy/pip/ct-pyspark-dev.txt
  ```
* If you encounter any issues with PySpark installation, check where it is installed using:
  ```bash
  pip show pyspark
  ```

  For a Conda environment, the installation path should look similar to:
  ```
  /home/{user}/conda/envs/{env-name}/lib/python3.10/site-packages
  ```

  If the path differs from the expected Conda environment path, check your Python path environment variable:
  ```bash
  echo $PYTHONPATH
  ```

  Ensure it is empty. If it is not, unset it using:
  ```bash
  unset PYTHONPATH
  ```

  Then, force reinstall PySpark to the correct environment location:
  ```bash
  pip install --force-reinstall --no-cache-dir pyspark=={version}
  ```

* Tip: Default name of the pyspark env is `ta-lib-pyspark-dev`. You can change it for your project.
    * For example: to make it as `env-myproject-pyspark`.
    * Open `tasks.py`
    * Set `ENV_PREFIX_PYSPARK = 'env-customer-pyspark'`
    * `dev` suffix will be added automatically after environment creation.

> **Note:**
> You can safely ignore the following warning while creating the Conda environment:
>
> `ERROR: Disabling PEP 517 processing is invalid: project specifies a build backend of setuptools.build_meta in pyproject.toml`
>
> This warning may occur because the `ta_lib` package is not installed in editable mode during environment creation.
>
> To install `ta_lib` properly, navigate to the root folder of **code-templates** and run:
>
> ```bash
> pip install -e .
> ```
>
> This will install the `ta_lib` package in editable mode and resolve the issue.

## Creating Conda Environment Without Specifying a Usecase

If you are creating a environment without specifying a usecase, you have the option to set a specific python
version. In this case the core packages will be installed from `ct-core-dev.txt`.
The below command will create a conda environment with python version `3.12`.

```
(base):~/<proj-folder>$ inv dev.setup-env --python-version=3.12
```
`python-version` parameter above is an optional parameter. By default it is set to `3.12` but it can take values of `3.10`, `3.11` or `3.12`.

## Creating Conda Environment Specifying a Usecase
When specifying a usecase, note that the `--python-version` option won't take effect. The environment will be created using the  `ct-core-dev.yml` and
the relevant usecase YAML file. The below command will create a conda environment for specific usecase.

```
(base):~/<proj-folder>$ inv dev.setup-env --usecase=<usecase>
```

Activate the environment first to install other addons. Keep the environment active for all the remaining commands in the manual.
```
(base):~/<proj-folder>$ conda activate ta-lib-dev
```

Install `invoke` and `pyyaml` in this env to be able to install the addons in this environment.
```
(ta-lib-dev):~/<proj-folder>$ pip install invoke
```

Now run all following command to install all the addons. Feel free to customize addons as suggested in the introduction.

```
(ta-lib-dev):~/<proj-folder>$ inv dev.setup-addon --formatting --jupyter --documentation --testing --extras --ts
```

You now should have a standalone conda python environment and installed code in the current repository along with all required dependencies.

* Get the installation info by running
    ```
    (ta-lib-dev):~/<proj-folder>$ inv dev.info
    ```

* Test your installation by running
    ```
    (ta-lib-dev):~/<proj-folder>$ inv test.val-env --usecase=<specific usecase>
    ```

We need to specify the usecase to validate the environment for core as well as usecase specific dependencies.

* This will just check the core setup, i.e, the env setup by inv dev.setup-env
* To check the addon installation in the conda env, we check it by specifying the specific addon like

    ```
    (ta-lib-dev):~/<proj-folder>$ inv test.val-env --formatting --jupyter --documentation --testing --extras --ts --pyspark
    ```
* You can specify which addon's installation you want to check here.

## Launching Jupyter Notebooks

- In order to launch a jupyter notebook locally in the web server, run

    ```
    (ta-lib-dev):~/<proj-folder>$ inv launch.jupyterlab
    ```
     After running the command, type [localhost:8080](localhost:8080) to see the launched JupyterLab.

> **Note:**
> * To use the **ebo** use case, install `patchwork` along with `invoke<2.0.0` and `fabric==2.7.3`.
> * The classification PySpark template notebooks have been updated with optimized Spark configurations (`spark.driver.memory` and `spark.sql.autoBroadcastJoinThreshold`) in `notebooks/reference/conf/config.yml`. These settings are pre-configured to improve stability and performance but can be adjusted based on your workload requirements.

# Setup environment using uv directly

If you are facing difficulties setting up the environment using the automated process (i.e., using the `invoke or task command`) or if the command is not accessible, you can use these manual steps.  This approach is
particularly useful when troubleshooting or in situations where automated setup is not feasible.
Follow the steps below to manually set up the environment.

## Step 1: Create virtual Environment
```
~/<proj-folder>$ uv venv <env_name> --python=<python_version>
```
Replace `<env_name>` with your specific environment name, and `<python_version>` with the desired python version (e.g., `3.10`, `3.11`, `3.12`).

### Create virtual Environment for PySpark
```
~/<proj-folder>$ uv venv <env_name> --python=<python_version>
```
Then install the dependencies required for PySpark.
```
~/<proj-folder>$ uv pip install .[ct_pyspark_dev]
```

## Step 2: Activate the Environment
```
~/<proj-folder>$ source <env_name>/bin/activate
```

## Step 3: Install Core Packages

Use the following command in the directory where `pyproject.toml` is located to install the core dependencies.

```
(<env_name>):~/<proj-folder>$ uv pip install
```

## Step 4: Install the ta_lib editable package
```
(<env_name>):~/<proj-folder>$ uv pip install -e <path_to_pyproject.toml>
```
if you are in the same level as the `pyproject.toml` file, you can use:
```
(<env_name>):~/<proj-folder>$ uv pip install -e .
```

## Step 5 (Optional): Install Additional Packages Based on Usecase

> **Note:** This should be run from the directory where the `pyproject.toml` file is located.

```
(<env_name>):~/<proj-folder>$ uv pip install .[ct_<usecase>_dev]
```
It takes a value of `tpo` or `mmx` or `ebo` or `rtm` or `reco`.
For example, to install packages for `mmx` usecase, use the following command:
```
(<env_name>):~/<proj-folder>$ uv pip install .[ct_mmx_dev]
```

### Step 6 (Optional): Install Additional Addons

> **Note:** This should be run from the directory where the `pyproject.toml` file is located.

For installing addons `bayesian_dev`, `extras`, `jupyter`, `tareg_dev`,  `ts_dev` use the below command
```
(<env_name>):~/<proj-folder>$ uv pip install .[addon_<addon-name>]
```
For example, to install `jupyter` addons, use the following command:
```
(<env_name>):~/<proj-folder>$ uv pip install .[addon_jupyter]
```

To install addons like `testing`, `formatting`, `documentation`, and `build`, use the following naming convention with `uv`:

- `testing` → `addon_testing`
- `formatting` → `addon_code_format`
- `documentation` → `addon_documentation`
- `build` → `pkg_build`

Use the command below, replacing `<addon-name>` with the appropriate group name:

```bash
(<env_name>):~/<proj-folder>$ uv pip install --group <addon-name>
```
# Setting Up Environment in Cloud
In a cloud environment invoke commands may not be accessible.
To set up the environment in a cloud setting, you can refer to the following link: [Cloud Environment Setup](https://tigeranalytics-code-templates.readthedocs-hosted.com/en/latest/code_templates/installation_setup.html)

# Frequently Asked Questions

The FAQ for code templates during setting up, testing, development and adoption phases are available
[here](https://tigeranalytics-code-templates.readthedocs-hosted.com/en/latest/faq.html)