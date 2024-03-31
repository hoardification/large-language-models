# Databricks notebook source
# MAGIC %run ../Includes/Classroom-Setup

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## Create documentation cells
# MAGIC Render cell as <a href="https://www.markdownguide.org/cheat-sheet/" target="_blank">Markdown</a> using the magic command: **`%md`**
# MAGIC
# MAGIC Below are some examples of how you can use Markdown to format documentation. Click this cell and press **`Enter`** to view the underlying Markdown syntax.
# MAGIC
# MAGIC
# MAGIC # Heading 1
# MAGIC ### Heading 3
# MAGIC > block quote
# MAGIC
# MAGIC 1. **bold**
# MAGIC 2. *italicized*
# MAGIC 3. ~~strikethrough~~
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC - <a href="https://www.markdownguide.org/cheat-sheet/" target="_blank">link</a>
# MAGIC - `code`
# MAGIC
# MAGIC ```
# MAGIC {
# MAGIC   "message": "This is a code block",
# MAGIC   "method": "https://www.markdownguide.org/extended-syntax/#fenced-code-blocks",
# MAGIC   "alternative": "https://www.markdownguide.org/basic-syntax/#code-blocks"
# MAGIC }
# MAGIC ```
# MAGIC
# MAGIC ![Spark Logo](https://files.training.databricks.com/images/Apache-Spark-Logo_TM_200px.png)
# MAGIC
# MAGIC | Element         | Markdown Syntax |
# MAGIC |-----------------|-----------------|
# MAGIC | Heading         | `#H1` `##H2` `###H3` `#### H4` `##### H5` `###### H6` |
# MAGIC | Block quote     | `> blockquote` |
# MAGIC | Bold            | `**bold**` |
# MAGIC | Italic          | `*italicized*` |
# MAGIC | Strikethrough   | `~~strikethrough~~` |
# MAGIC | Horizontal Rule | `---` |
# MAGIC | Code            | ``` `code` ``` |
# MAGIC | Link            | `[text](https://www.example.com)` |
# MAGIC | Image           | `![alt text](image.jpg)`|
# MAGIC | Ordered List    | `1. First items` <br> `2. Second Item` <br> `3. Third Item` |
# MAGIC | Unordered List  | `- First items` <br> `- Second Item` <br> `- Third Item` |
# MAGIC | Code Block      | ```` ``` ```` <br> `code block` <br> ```` ``` ````|
# MAGIC | Table           |<code> &#124; col &#124; col &#124; col &#124; </code> <br> <code> &#124;---&#124;---&#124;---&#124; </code> <br> <code> &#124; val &#124; val &#124; val &#124; </code> <br> <code> &#124; val &#124; val &#124; val &#124; </code> <br>|
# MAGIC
# MAGIC - Magic Commands <a href="https://docs.databricks.com/notebooks/notebooks-use.html#language-magic" target="_blank">magic commands</a>: **`%python`**, **`%sql`**, **`%md`**, **`%fs`**, **`%sh`**, **`%pip`**

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ## Reading data
# MAGIC
# MAGIC When you ran the **Setup** cell at the top of the notebook, some variables were created for you. One of the variables is `DA.paths.datasets` which is the path to datasets.
# MAGIC
# MAGIC One such dataset is located at **`{DA.paths.datasets}/news/labelled_newscatcher_dataset.csv`**. Let's use `pandas` to read that csv file.

# COMMAND ----------

import pandas as pd

# Specify the location of the csv file
csv_location = f"{DA.paths.datasets}/news/labelled_newscatcher_dataset.csv"
# Read the dataset
newscatcher = pd.read_csv(csv_location, sep=";")
# Display the datset
newscatcher

# COMMAND ----------

# MAGIC %md
# MAGIC `matplotlib` to plot aggregate data from dataset.

# COMMAND ----------

import matplotlib.pyplot as plt

# Count how many articles exist per topic
newscatcher_counts_by_topic = (
    newscatcher
    .loc[:,["topic","title"]]
    .groupby("topic")
    .agg("count")
    .reset_index(drop=False)
)

# Create a bar plot
plt.bar(newscatcher_counts_by_topic["topic"],height=newscatcher_counts_by_topic["title"])
plt.xticks(rotation=45)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC The `display()` command will pretty-print a large variety of data types, including Apache Spark DataFrames or Pandas DataFrames.
# MAGIC
# MAGIC It will also allow you to make visualizations without writing additional code. For example, after executing the below command click the `+` icon in the results to add a Visualization. Select the **Bar** visualization type and click "Save".

# COMMAND ----------

display(newscatcher_counts_by_topic)
