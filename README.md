# SmartFleet Predict: AI-Driven Vehicle Maintenance System

## Project Overview

**SmartFleet Predict** is an AI-driven solution designed to revolutionize vehicle fleet management by predicting maintenance needs before critical failures occur. This system leverages machine learning to analyze various vehicle parameters, enabling proactive maintenance scheduling, thereby minimizing downtime, reducing operational costs, and extending vehicle lifespan.

This project was developed based on a problem statement requiring the creation of a predictive maintenance model and an interactive web dashboard to help fleet managers optimize operations.

## Problem Statement

Public transportation fleets frequently face unexpected mechanical failures, leading to route delays, increased operational costs, and customer dissatisfaction. Most current maintenance systems rely on periodic manual inspections rather than real-time, data-driven insights. This inefficiency results in unplanned breakdowns and poor resource utilization.

## Solution

This project addresses the problem by developing a machine learning-based predictive maintenance system. It analyzes historical vehicle telemetry and maintenance data to forecast component failures in advance.

The final system includes an XGBoost classification model and an interactive Streamlit dashboard. Fleet managers can input a vehicle's current operational data and receive an instant prediction on whether the vehicle requires maintenance, helping them prioritize inspections and reduce downtime.

## Features

* **Data-Driven Prediction:** Uses machine learning to forecast vehicle breakdown risk.
* **Cost Reduction:** Aims to reduce maintenance costs and vehicle downtime.
* **Operational Efficiency:** Designed to improve overall fleet performance.
* **Interactive Dashboard:** A web-based dashboard for real-time monitoring and predictions.
* **Data-Leakage & Imbalance Handled:** The model was carefully built to remove data leakage (e.g., diagnostic results) and manage a severe class imbalance for robust, real-world predictions.

## Dashboard Preview

Here is a preview of the final Streamlit dashboard, which allows managers to input vehicle data and receive an instant prediction.

![SmartFleet Predict Dashboard Screenshot](https://github.com/Ayush-Shrivas/SmartFleet-Predict-AI-Driven-Vehicle-Maintenance-System/blob/main/Demo/Website%20Demo.png)

## Video Demonstration

Click the image below to watch a full video walkthrough of the project, from running the app to performing a live prediction.

[![Watch the video](dashboard-preview.png)](https://github.com/Ayush-Shrivas/SmartFleet-Predict-AI-Driven-Vehicle-Maintenance-System/blob/main/Demo/Jupyter%20Souce%20Code%20Screen%20Record.mp4)

**(Remember to replace `https://www.youtube.com/watch?v=YOUR_VIDEO_ID` with the actual link to your uploaded YouTube video. You can use the same `dashboard-preview.png` as the clickable thumbnail!)**

## Tech Stack

The tools used in this project were based on the problem statement requirements:

| Category | Tools Used |
| :--- | :--- |
| **Programming Language** | Python |
| **Data Handling & Analysis** | Pandas, NumPy |
| **Modeling & Machine Learning** | Scikit-Learn, XGBoost |
| **Visualization** | Matplotlib, Seaborn |
| **Dashboard / Web Deployment** | Streamlit |
| **Data Source** | Kaggle Vehicle Maintenance Dataset |
| **Version Control** | Git, GitHub |
| **Development Environment** | Jupyter Notebook / VS Code |
| **Model Serialization** | Joblib |

## Model Performance

After identifying and removing data-leaking features (like `Brake_Condition` and `Reported_Issues`, which are results of an inspection, not predictors of one), the model had to be trained on a dataset with a weak signal and significant class imbalance.

The final balanced XGBoost model achieved a realistic and useful performance:

| Class | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- |
| **No Maintenance (0)** | 0.26 | 0.50 | 0.34 |
| **Needs Maintenance (1)** | 0.85 | 0.65 | 0.74 |
| **Weighted Avg** | 0.74 | 0.63 | 0.66 |

**Key Insight:** The model successfully learned to identify 50% of the vehicles that **did not** need maintenance (Recall=0.50 for Class 0). This is highly valuable for a fleet manager, as it allows them to confidently reduce unnecessary inspections, saving time and money.

## How to Run Locally

Follow these steps to set up and run the project on your local machine.

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/Ayush-Shrivas/SmartFleet-Predict-AI-Driven-Vehicle-Maintenance-System.git](https://github.com/Ayush-Shrivas/SmartFleet-Predict-AI-Driven-Vehicle-Maintenance-System.git)
    cd "SmartFleet Predict AI-Driven Vehicle Maintenance System"
    ```

2.  **Set Up Virtual Environment:**
    ```bash
    python -m venv venv
    .\venv\Scripts\activate # On Windows
    # source venv/bin/activate # On macOS/Linux
    ```

3.  **Install Dependencies:**
    *(First, create a `requirements.txt` file in your project folder and paste the package names below into it.)*
    ```
    pandas
    numpy
    scikit-learn
    xgboost
    matplotlib
    seaborn
    streamlit
    joblib
    imbalanced-learn
    ```
    Now, run the installation:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download Data:**
    * Download the dataset from [Kaggle: Vehicle Maintenance Dataset](https://www.kaggle.com/datasets/chavindudulaj/vehicle-maintenance-data).
    * Create a `data` folder in your project directory.
    * Place the downloaded CSV file into the `data/` folder.

5.  **Run the Jupyter Notebook:**
    * Launch the `01-EDA.ipynb` notebook to see the full data exploration, preprocessing, data leakage discovery, and model training process.
    * Running this notebook will generate the final `smartfleet_model.joblib` file.
    ```bash
    jupyter lab
    ```

6.  **Run the Streamlit Dashboard:**
    * Ensure the `smartfleet_model.joblib` file is in your root project folder.
    * Run the following command in your terminal:
    ```bash
    streamlit run app.py
    ```
    Your application will open in your web browser at `http://localhost:8501`.
