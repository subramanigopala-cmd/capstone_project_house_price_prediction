## Environment & Complete Project Execution

This project uses two separate Python environments because the project contains both the standard housing-price prediction workflow and additional TA-Regression experiments.

### 1. Python Environments

Two environments are used intentionally:

| Environment        | Purpose                                                                                                                                                                     |
| ------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `ta-lib-dev`       | Main development environment for the housing project, production pipeline, notebooks, unit tests, code-quality checks, Radon complexity analysis, and Sphinx documentation. |
| `ta-lib-tareg-dev` | Dedicated TA-Regression environment used for TA-Regression experiments in Notebook 04, including ElasticNet and LinearMixedEffects models.                                  |

The two environments are **not interchangeable for every command** because the required dependencies differ.

### 2. Main Development Environment — `ta-lib-dev`

Activate the main environment:

```bash
source ta-lib-dev/bin/activate
```

Use this environment for the normal project workflow.

#### Run the production pipeline

```bash
./ta-lib-dev/bin/python -m production.cli job run --job-id all
```

The production workflow performs:

1. Housing data cleaning
2. Stratified train/test split
3. Housing feature engineering
4. Feature preprocessing
5. Random Forest model training
6. Model scoring

Generated artifacts include:

```text
artifacts/curated_columns.joblib
artifacts/features.joblib
artifacts/train_pipeline.joblib
data/score/housing/output.parquet
```

#### Run unit tests

```bash
task test.unittest
```

The housing transformer tests verify:

* Forward transformation
* Inverse transformation
* Handling of negative values
* Reduction of large numeric values
* scikit-learn estimator compatibility

#### Run code-quality checks

```bash
task test.qc
```

This runs:

* Black
* isort
* flake8

Third-party/vendor code inherited from the original template is excluded from the applicable formatting/quality checks where necessary.

#### Run Radon complexity analysis

Radon is installed in `ta-lib-dev` and is therefore run explicitly using this environment:

```bash
./ta-lib-dev/bin/python -m radon cc src/ta_lib/ -s -a
```

The project currently reports:

```text
2539 blocks (classes, functions, methods) analyzed.
Average complexity: A (3.7136667979519498)
```

The average complexity is graded **A**.

> Note: `task test.complexity` uses the currently active Python environment. Therefore, when `ta-lib-tareg-dev` is active, use the explicit `ta-lib-dev` command above, or activate `ta-lib-dev` before running the task.

#### Build the documentation

Sphinx is installed in `ta-lib-dev`.

Activate the main environment:

```bash
source ta-lib-dev/bin/activate
```

Then build the documentation:

```bash
task build.docs
```

Alternatively, the Sphinx executable can be called explicitly:

```bash
./ta-lib-dev/bin/sphinx-build -b html docs/source docs/build/html
```

The generated HTML documentation is available under:

```text
docs/build/html/
```

A successful build ends with:

```text
build succeeded.
The HTML pages are in docs/build/html.
```

### 3. TA-Regression Environment — `ta-lib-tareg-dev`

The TA-Regression environment is maintained separately because some TA-Regression dependencies are different from the main development environment.

Activate it with:

```bash
source ta-lib-tareg-dev/bin/activate
```

This environment is primarily used for **Notebook 04 — TA-Regression Model Experimenting**.

Notebook 04 evaluates additional regression approaches, including:

* TA-Regression ElasticNet
* TA-Regression LinearMixedEffects
* Random Forest benchmark

The Bayesian/NumPyro experiment is not used in the final workflow because the available Bambi/NumPyro implementation does not support the required inference method in this environment.

### 4. Running Notebook 04

From the notebook directory:

```bash
cd notebooks/reference
```

Execute Notebook 04 using the TA-Regression environment:

```bash
rm -f 04_ta_reg_model_experimenting_executed.ipynb

../../ta-lib-tareg-dev/bin/jupyter nbconvert \
  --to notebook \
  --execute 04_ta_reg_model_experimenting.ipynb \
  --output 04_ta_reg_model_experimenting_executed.ipynb \
  --ExecutePreprocessor.timeout=1200
```

A successful execution creates:

```text
notebooks/reference/04_ta_reg_model_experimenting_executed.ipynb
```

### 5. Notebook Execution

The project contains four reference notebooks:

```text
notebooks/reference/
├── 01_data_discovery.ipynb
├── 02_data_processing.ipynb
├── 03_model_experimenting.ipynb
└── 04_ta_reg_model_experimenting.ipynb
```

#### Notebook 01 — Data Discovery

Explores the California housing dataset, including:

* Dataset structure
* Data types
* Missing values
* Duplicate records
* Numerical distributions
* Outliers
* Correlations
* Geographic relationships
* Housing value by ocean proximity

#### Notebook 02 — Data Processing

Performs the housing data-processing workflow and produces:

```text
cleaned/housing
processed/housing
train/housing/features
train/housing/target
test/housing/features
test/housing/target
```

The feature set includes the original housing variables and engineered variables such as:

```text
rooms_per_household
bedrooms_per_room
population_per_household
```

#### Notebook 03 — Model Experimenting

Compares several regression models and performs Random Forest tuning.

The final tuned Random Forest configuration is:

```text
max_depth=20
max_features=0.5
min_samples_leaf=2
n_estimators=215
n_jobs=-1
random_state=0
```

The trained model is saved as:

```text
artifacts/housing_final_model.joblib
```

#### Notebook 04 — TA-Regression Model Experimenting

Uses the dedicated `ta-lib-tareg-dev` environment to experiment with TA-Regression models and compare their performance with the Random Forest benchmark.

### 6. Recommended Complete Validation Sequence

After making project changes, the following sequence provides a complete validation of the project.

#### Step 1 — Main environment

```bash
source ta-lib-dev/bin/activate
```

#### Step 2 — Code-quality checks

```bash
task test.qc
```

#### Step 3 — Complexity analysis

```bash
./ta-lib-dev/bin/python -m radon cc src/ta_lib/ -s -a
```

#### Step 4 — Production pipeline

```bash
./ta-lib-dev/bin/python -m production.cli job run --job-id all
```

#### Step 5 — TA-Regression environment

```bash
source ta-lib-tareg-dev/bin/activate
```

Run Notebook 04 from:

```bash
cd notebooks/reference
```

#### Step 6 — Unit tests

```bash
cd /mnt/c/capstone_project_house_price_prediction
task test.unittest
```

#### Step 7 — Documentation

Switch back to the main development environment:

```bash
source ta-lib-dev/bin/activate
```

Build the documentation:

```bash
task build.docs
```

### 7. Expected Validation Results

A successful project validation should produce results similar to:

```text
Quality Control
---------------
Black       PASS
isort       PASS
flake8      PASS

Complexity
----------
Average complexity: A

Unit Tests
----------
4 passed

Documentation
-------------
build succeeded
```

The production scoring output is generated at:

```text
data/score/housing/output.parquet
```

and the trained production pipeline is generated at:

```text
artifacts/train_pipeline.joblib
```

### 8. Important Environment Note

The two environments are intentionally retained rather than merged.

`ta-lib-dev` is the **primary project environment** and contains the tools required for:

* Production execution
* Standard notebooks
* Unit testing
* Quality control
* Radon
* Documentation

`ta-lib-tareg-dev` is the **specialized TA-Regression environment** and is retained for the additional regression experiments in Notebook 04.

When executing a command, always use the environment specified by this README to avoid dependency-related errors.

### 9. Generated Files

The following directories may contain generated runtime or build artifacts:

```text
docs/build/
.pytest_cache/
__pycache__/
artifacts/
data/score/
logs/
```

Before committing the project, review `git status` and ensure that only intended project files, configuration, notebooks, documentation, tests, and required artifacts are included.
