# 示例数据说明

本目录下的数据为 SinoPath 实验框架提供了一个最小可行示例，用于演示框架的基本功能。

## 文件结构

-   `learning_outcomes.csv`: 包含所有教育体系中的学习成果（Learning Outcomes, LOs）。
-   `modules.csv`: 包含所有教育体系中的课程模块（Modules）。

## 数据模式

### `learning_outcomes.csv`

| 字段名             | 类型    | 描述                                     | 示例值         |
| ------------------ | ------- | ---------------------------------------- | -------------- |
| `lo_id`            | `str`   | 学习成果的唯一标识符。                   | `AL001_LO01`   |
| `module_id`        | `str`   | 该学习成果所属的模块 ID，关联 `modules.csv`。 | `AL001`        |
| `system`           | `str`   | 该学习成果所属的教育体系名称。           | `A-level`      |
| `description`      | `str`   | 对学习成果的具体文本描述。               | `Understand the principles of differentiation` |
| `difficulty_level` | `int`   | 难度等级，范围 0~10：0=无需前置知识；10=高度专门无法在通识教学中完成。 | `3`            |
| `module_weight`    | `float` | （可选）模块在所属课程中的权重。若缺失则按课程内均分并归一化为和=1。 | `0.25`         |
| `weight`           | `float` | 该学习成果在所属模块中的重要性权重，0 到 1。 | `0.3`          |

### `modules.csv`

| 字段名        | 类型    | 描述                           | 示例值        |
| ------------- | ------- | ------------------------------ | ------------- |
| `module_id`   | `str`   | 模块的唯一标识符。             | `AL001`       |
| `course_id`   | `str`   | 模块所属的课程 ID。            | `AL-MATH`     |
| `module_name` | `str`   | 模块的名称。                   | `Calculus I`  |
| `system`      | `str`   | 模块所属的教育体系名称。       | `A-level`     |
| `topic_area`  | `str`   | 模块所属的主题领域（可选）。   | `Mathematics` |

## 最小可行示例说明

为了保证框架能够顺利运行，您的自定义数据必须满足以下最小要求：

-   **数据完整性**: 每个文件都不能有缺失的 `lo_id`, `module_id`, `system`, `description`。
-   **关联性**: `learning_outcomes.csv` 中的 `module_id` 必须能在 `modules.csv` 中找到对应的条目。
-   **体系存在**: 在执行映射时，指定的 `source_system` 和 `target_system` 必须在两个文件中都存在。
-   **数据量**: 每个待映射的体系中，至少包含一个模块，且每个模块至少包含一个学习成果。
-   **模块权重**: 若提供 `module_weight`，请确保一个课程下所有模块权重和为 1；否则框架自动均分。
