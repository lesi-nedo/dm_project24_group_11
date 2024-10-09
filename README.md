# 🚴‍♂️ Data Mining Project: Professional Cycling Analysis

## 🎯 Project Overview

This project focuses on implementing and analyzing a data mining pipeline centered around professional cycling races. Our primary objective is to gain deep insights into the performance and characteristics of cyclists and races through various advanced data mining techniques.

### 🔍 Key Objectives

- Uncover hidden patterns in cycling race data
- Analyze cyclist performance across different race conditions
- Identify factors that contribute to race outcomes
- Explore relationships between cyclist attributes and race results

## 📊 Dataset Description

Our analysis leverages a comprehensive dataset that includes detailed information about individual races and participating cyclists.

### 📈 Data Structure

Two main tables:
1. 🏁 **Races**: Details about individual races
2. 🚴‍♀️ **Cyclists**: Information about participating cyclists

### 🏷️ Key Features

| Category | Features |
|----------|----------|
| Race Attributes | Identifiers, names, points, lengths, profiles |
| Cyclist Attributes | Weight, height, team affiliation |
| Performance Metrics | Finish positions, time deltas |

## 🛠️ Project Structure

The project is divided into several interconnected tasks, each contributing to a comprehensive analysis of the cycling data.

### 1️⃣ Data Understanding
- Explore data distributions
- Identify potential data quality issues
- Visualize relationships between features

### 2️⃣ Data Transformation
- Engineer novel features (e.g., cyclist performance on different terrains)
- Detect and handle outliers
- Prepare data for advanced analysis

### 3️⃣ Clustering Analysis
- Apply various clustering algorithms:
  - K-means
  - Density-based clustering
  - Hierarchical clustering
- Identify and describe distinct groups within the cycling ecosystem

## 📝 Reporting Requirements

A thorough report is essential, detailing:

- 🤔 Motivations behind each analysis choice
- 🔬 Methodologies employed and their justifications
- 💡 Insights gained from the analyses
- ⚖️ Limitations of the results and potential areas for further investigation

## 🏆 Project Goals

By completing this data mining project, we aim to:

1. 🎓 Demonstrate practical application of data mining techniques
2. 🏋️‍♂️ Showcase the potential of data-driven approaches in sports analytics
3. 🧠 Develop a deeper understanding of professional cycling dynamics
4. 📊 Create valuable insights for teams, athletes, and race organizers

## 💻 Technical Implementation

- Utilize Python for data processing and analysis
- Leverage libraries such as pandas, scikit-learn, and matplotlib
- Implement custom algorithms as needed for specialized analyses

## 🌟 Expected Outcomes

- Identification of key factors influencing cycling race outcomes
- Cyclist performance profiles across various race conditions
- Insights into team strategies and their effectiveness
- Potential predictive models for race results or cyclist performance

---

🚴‍♂️💨 Let's dive into the exciting world of professional cycling data and uncover the hidden stories within the numbers! 🏆📊




# 🚴‍♂️ Cycling, Cycling, and More Cycling: Data Mining Project 2024/25

## 📋 Assignment Overview

This assignment focuses on implementing and analyzing a data mining pipeline, as covered during the course.

### 🎯 Deliverables

1. 📄 A report (max 25 pages, including figures)
2. 💻 Source code (to be submitted via GitHub)

**Note**: Both components contribute to the final grade.

### 👥 Team Formation

Work in teams of **3 students**.

### 🛠️ Tools

Utilize tools and techniques presented during the course.

---

## 📊 Dataset Description

### 🌍 Domain

The dataset comprises professional cycling races spanning several years.

### 📈 Data Structure

Two main tables:
1. 🚴‍♀️ Cyclists
2. 🏁 Races

### 🏷️ Features

| Feature | Description | Example |
|---------|-------------|---------|
| _url | Race identifier | tour-de-france/1978/stage-6 |
| name | Race name | Tour de France |
| points | Race prestige points | 100 |
| uci_points | Alternative race points | 100 |
| length | Race length (meters) | 162000 |
| climb_total | Total climb (meters) | 3512 |
| profile | Race difficulty profile | flat, hilly, mountainous, high mountains |
| startlist_quality | Participant strength | - |
| date | Race date | - |
| position | Cyclist's finish position | 3 |
| cyclist | Cyclist's ID | sean-kelly |
| is_X | Alternative surface indicator | - |
| cyclist_team | Cyclist's team | visma-lease-a-bike-2024 |
| delta | Time behind winner (seconds) | - |
| weight | Cyclist's weight (kg) | 64.2 |
| height | Cyclist's height (cm) | 178 |

---

## 📝 Report Guidelines

### General Principles

For each analysis, include:

- 🎯 **Motivations**: Explain your choices
- 🔍 **Thorough analysis**: Consider various settings and hyperparameters
- 💡 **Observations**: Share insights gained
- ⚠️ **Limitations**: Discuss the strength of your findings

### Tasks

#### 1️⃣ Data Understanding (10 points)

- Assess data quality
- Analyze data distribution
- Explore relationships between features

#### 2️⃣ Data Transformation (20 points)

1. 🛠️ Feature engineering / novel feature definition
2. 🔍 Outlier detection
3. 🔄 Revised data understanding (including new features and outlier considerations)

**Ideas for feature engineering:**
- Segment the racing season
- Analyze cyclists on different terrains/profiles
- Study cyclists at various ages

#### 3️⃣ Clustering (30 points + 2 bonus points)

- Identify and describe instance groups
- Consider cyclists, races, or both
- Use features defined in the data transformation task

**Required clustering algorithms:**
- K-means clustering
- Density-based clustering
- Hierarchical clustering

**Bonus:** Experiment with additional clustering algorithms for up to 2 extra points.

---

🏆 Good luck with your cycling data mining project! 🚴‍♂️💨
