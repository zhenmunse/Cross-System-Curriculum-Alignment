# Data Specifications

This directory contains the sample dataset used for the experimental validation of the framework. These files serve as a reference implementation to demonstrate the core functionality of the alignment pipeline.

## File Structure

- **learning_outcomes.csv**: Contains atomic Learning Outcomes (LOs) for all educational systems.
- **modules.csv**: Defines the curriculum modules and their hierarchical relationships.

## Data Schema

### learning_outcomes.csv

| **Field Name**       | **Type** | **Description**                                              | **Example**                                    |
| -------------------- | -------- | ------------------------------------------------------------ | ---------------------------------------------- |
| **lo_id**            | `str`    | Unique identifier for the learning outcome.                  | `AL001_LO01`                                   |
| **module_id**        | `str`    | Foreign key linking the outcome to a specific entry in `modules.csv`. | `AL001`                                        |
| **system**           | `str`    | Name of the educational system (e.g., A-level, University).  | `A-level`                                      |
| **description**      | `str`    | Textual description of the learning outcome.                 | `Understand the principles of differentiation` |
| **difficulty_level** | `int`    | Cognitive difficulty score (0–10) based on Bloom's Taxonomy. | `3`                                            |
| **module_weight**    | `float`  | (Optional) Relative weight of the module within its parent course. | `0.25`                                         |
| **weight**           | `float`  | Relative importance of the LO within its parent module (0.0 to 1.0). | `0.3`                                          |

### modules.csv

| **Field Name**  | **Type** | **Description**                                              | **Example**   |
| --------------- | -------- | ------------------------------------------------------------ | ------------- |
| **module_id**   | `str`    | Unique identifier for the curriculum module.                 | `AL001`       |
| **course_id**   | `str`    | Identifier for the parent course or academic unit.           | `AL-MATH`     |
| **module_name** | `str`    | Descriptive name of the module.                              | `Calculus I`  |
| **system**      | `str`    | Name of the educational system.                              | `A-level`     |
| **topic_area**  | `str`    | (Optional) Conceptual domain or disciplinary area of the module. | `Mathematics` |

## Requirements for Custom Datasets

To ensure successful execution and reproducibility of the alignment analysis, custom datasets must satisfy the following technical requirements:

- **Relational Integrity**: Every `module_id` referenced in `learning_outcomes.csv` must possess a corresponding record in `modules.csv`.
- **Data Completeness**: Essential fields including identifiers, system names, descriptions, and difficulty levels must be free of null or missing values.
- **System Consistency**: The specified `source_system` and `target_system` used during execution must be present in the `system` columns of both data files.
- **Curricular Density**: Each system targeted for mapping must contain at least one module, and each module must contain at least one learning outcome.
- **Weight Normalization**: If the `module_weight` column is provided, the sum of weights for all modules within a single course should ideally equal 1.0. If these values are missing or invalid, the framework automatically applies a normalized uniform distribution across all modules.
