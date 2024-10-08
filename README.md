# Data Mining Project


This project focuses on implementing and analyzing a data mining pipeline centered around professional cycling races. The primary objective is to gain insights into the performance and characteristics of cyclists and races through various data mining techniques. By leveraging the provided dataset, which includes detailed information about individual races and participating cyclists, we aim to uncover patterns and relationships that can inform our understanding of the sport.

The dataset comprises two main tables: cyclists and races. The cyclists table contains details about individual races, while the races table provides information about the cyclists participating in those races. Key features of the dataset include race identifiers, names, points, lengths, profiles, and cyclist-specific attributes such as weight and height.

The project is structured into several tasks, each contributing to a comprehensive analysis of the data. These tasks include data understanding, data transformation, and clustering. The ultimate goal is to identify and describe groups of instances using clustering algorithms, providing a deeper understanding of the dynamics within professional cycling races.

In addition to the technical analysis, the project requires a thorough report that explains the motivations behind the chosen analysis, the methodologies employed, and the insights gained. The report should also discuss the limitations of the results and provide clear justifications for the choices made throughout the project.

By completing this project, we aim to demonstrate the application of data mining techniques in a real-world context, showcasing the potential of data-driven approaches in sports analytics.

****


🚴‍♂️ **Cycling, Cycling, and More Cycling** 🚴‍♂️
📅 2024/25

**Assignment:**
* Implement and analyze the data mining pipeline as discussed in class.

**Deliverables:**
1. A report of maximum 25 pages (including figures).
2. Source code to create the analysis (delivered via GitHub).

**Teams:** Work in teams of 3 students.

**Tools:** Use tools and techniques presented in the course.

**Dataset Description:**

**Domain:** Professional cycling races.

**Data:**
* **cyclists** table: Details about individual races.
* **races** table: Details about participating cyclists.

**Features:**

| Feature | Description | Example |
|---|---|---|
| url | Identifier of the race | tour-de-france/1978/stage-6 |
| name | Name of the race | Tour de France |
| points | Points assigned to the race | 100 |
| uci_points | Alternative points | 100 |
| length | Length of the race | 162000 |
| climb_total | Total meters climbed | 3512 |
| profile | Race profile | flat, hilly, mountainous, high mountains |
| startlist_quality | Strength of participants | ... |
| date | Race date | ... |
| position | Finish position | 3 |
| cyclist | ID of the cyclist | sean-kelly |
| is_X | Is the race run on a X surface? | ... |
| cyclist_team | Team the cyclist belongs to | visma-lease-a-bike-2024 |
| delta | Seconds after the first-placed | ... |
| weight | Weight of the cyclist | 64.2 |
| height | Height of the cyclist | 178 |

**Report Guidelines:**

* **Motivations:** Explain why you chose this analysis.
* **Thorough Analysis:** Consider various hyperparameters and justify your choices.
* **Observations:** Share insights and information gained.
* **Limitations:** Discuss the strengths and weaknesses of your results.

**Tasks:**

**Task 1: Data Understanding (10 points)**
* Assess data quality, distribution, and relationships.

**Task 2: Data Transformation (20 points)**
* Feature engineering and/or novel feature definition.
* Outlier detection.
* Revamped data understanding.

**Task 3: Clustering (30 points + 2 bonus points)**
* Identify and describe groups of instances using clustering algorithms.
* Consider clustering cyclists, races, or both.
* Experiment with different clustering algorithms (k-means, density-based, hierarchical).
* Compare and analyze different clusterings.

**Additional Notes:**

* Use emojis to enhance readability and engagement.
* Provide clear explanations and justifications for your choices.
* Feel free to experiment with additional clustering algorithms.