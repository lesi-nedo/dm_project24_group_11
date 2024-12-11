# 🚴‍♂️ Data Mining Project: Professional Cycling Analysis

## 🎯 Project Overview

This project focuses on implementing and analyzing a data mining pipeline centered around professional cycling races. Our primary objective is to gain deep insights into the performance and characteristics of cyclists and races through various advanced data mining techniques.

### 🔍 Key Objectives

- Implement and analyze the data mining pipeline for professional cycling races
- Uncover hidden patterns and relationships in cycling race data
- Analyze cyclist performance across different race conditions and attributes
- Identify factors that contribute to race outcomes
- Develop predictive models for race results
- Provide explanations for predictive models using interpretability techniques

## 🚀 Project Overview

This project is an implementation and analysis of the data mining pipeline as highlighted during the course. The deliverables include:

- 📄 A report of maximum 25 pages (including figures)
- 💻 Source code used for the analysis (delivered through GitHub)

**Teams:** The project is to be completed in teams of three students.

**Tools:** The project must use tools and techniques presented during the course.

## 📊 Dataset Description

The dataset comprises a set of professional cycling races, spanning several years, and includes detailed information about individual races and participating cyclists.

### 📈 Data Structure

Two main tables:

1. 🏁 **Races**: Details about individual races
2. 🚴‍♀️ **Cyclists**: Information about participating cyclists

### 🏷️ Key Features

| Category            | Features            | Description                                                               | Example                          |
|---------------------|---------------------|---------------------------------------------------------------------------|----------------------------------|
| **Race Attributes** | `_url`              | Identifier of the race                                                    | `tour-de-france/1978/stage-6`    |
|                     | `name`              | Name of the race                                                          | `Tour de France`                 |
|                     | `points`            | Points assigned to the race (prestige indicator)                          | `100`                            |
|                     | `uci_points`        | Alternative points assigned to the race                                   | `100`                            |
|                     | `length`            | Length of the race in meters                                              | `162000`                         |
|                     | `climb_total`       | Total meters climbed during the race                                      | `3512`                           |
|                     | `profile`           | Race profile (e.g., flat, hilly, mountainous)                             | `mountainous`                    |
|                     | `startlist_quality` | Strength of the participants                                              | `1241`                           |
|                     | `date`              | Race date                                                                 | `2021-07-14`                     |
| **Cyclist Attributes** | `cyclist`         | ID of the cyclist                                                         | `sean-kelly`                     |
|                     | `position`          | Finish position of the cyclist                                            | `3`                              |
|                     | `cyclist_team`      | Team the cyclist belongs to                                               | `visma-lease-a-bike-2024`        |
|                     | `weight`            | Weight of the cyclist in kg                                               | `64.2`                           |
|                     | `height`            | Height of the cyclist in cm                                               | `178`                            |
| **Performance Metrics** | `delta`          | Time difference in seconds after the first place                          | `45`                             |
| **Other Attributes** | `is_X`             | Indicates if the race includes a specific surface (e.g., gravel)          | `True`                           |

## 🛠️ Project Structure

The project is divided into several interconnected tasks:

### 1️⃣ Data Understanding (10 points)

- **Assess Data Quality:**
  - Identify missing or incorrect values
  - Analyze data distribution
  - Examine relationships between features

### 2️⃣ Data Transformation (20 points)

1. **Feature Engineering and Novel Feature Definition:**
   - Improve data quality by addressing missing/incorrect values
   - Engineer new features involving cyclists, races, teams, etc.
   - Examples:
     - Segment the racing season
     - Analyze cyclist performance on different terrains or race profiles
     - Study cyclists at different ages
2. **Outlier Detection:**
   - Identify and handle outliers in the dataset
3. **Revised Data Understanding:**
   - Reassess data with new features and consider outlier impacts
   - Update previous analyses to reflect changes

### 3️⃣ Clustering Analysis (30 points + 2 bonus points)

- **Objective:**
  - Identify and describe groups within the data
  - Consider clustering cyclists, races, or both
- **Algorithms to Apply:**
  - K-means clustering
  - Density-based clustering
  - Hierarchical clustering
- **Additional Work:**
  - Experiment with additional clustering algorithms for up to 2 bonus points
- **Deliverables:**
  - Observations and comparisons of different clustering results

### 4️⃣ Predictive Modeling (30 points)

- **Goal:**
  - Develop models to predict if a cyclist will finish in the top 20 positions
- **Data Split:**
  - Use races prior to 2022 for training
  - Use races from 2022 onward as the test set
- **Task Details:**
  - Formulate as a binary classification problem
  - Explore various algorithms and hyperparameters
  - Validate models using appropriate metrics
  - Compare models to identify the best performers

### 5️⃣ Explanation (30 points)

- **Provide Explanations for Predictive Models:**
  - Focus on:
    - Feature importance
    - Rule-based explanations
    - Counterfactual instances
- **Analysis:**
  - Evaluate explanations for fidelity and complexity
  - Compare model insights with findings from previous tasks
    - Did the model capture patterns identified earlier?
    - Were new or unexpected patterns discovered?
    - Address any nonsensical patterns identified

## 📝 Reporting Requirements

A thorough report is essential, detailing:

- **Motivations:**
  - Justify the choice of analyses and feature engineering
- **Methodologies:**
  - Describe the methods and algorithms used
  - Explain parameter choices and settings
- **Observations:**
  - Present insights gained from each analysis
- **Limitations:**
  - Discuss the strength of analytical results and observations
  - Highlight potential areas for further investigation

## 🏆 Project Goals

By completing this data mining project, we aim to:

1. 🎓 Demonstrate practical application of data mining techniques
2. 🏋️‍♂️ Showcase the potential of data-driven approaches in sports analytics
3. 🧠 Develop a deeper understanding of professional cycling dynamics
4. 📊 Create valuable insights for teams, athletes, and race organizers

## 💻 Technical Implementation

- Utilize Python for data processing and analysis
- Leverage libraries such as pandas, scikit-learn, matplotlib, and others presented during the course
- Implement custom algorithms as needed for specialized analyses

## 🌟 Expected Outcomes

- Identification of key factors influencing cycling race outcomes
- Cyclist performance profiles across various conditions
- Insights into team strategies and their effectiveness
- Predictive models for race results or cyclist performance
- Interpretations and explanations of predictive models
- Interactive visualizations to present key findings


---

🚴‍♂️💨 Let's dive into the exciting world of professional cycling data and uncover the hidden stories within the numbers! 🏆📊
