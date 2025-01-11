# :mag: XplainIt - Explainable AI Dashboard

A Streamlit application for model interpretability using SHAP (SHapley Additive exPlanations) and LIME (Local Interpretable Model-Agnostic Explanations).

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://xplainit-shquvhbhpwm3imswdpyzhh.streamlit.app/)

## :sparkles: Features

- **SHAP Analysis**: Comprehensive global and local interpretability using SHAP.
  - SHAP summaries, dependence plots, and positive/negative contributions.
- **LIME Analysis**: Local interpretability of specific data points.
  - Original and edited visualizations.
- **Model Comparison**: Evaluate multiple machine learning models with key metrics.
- **Customizable Missing Data Handling**: Flexible strategies to handle missing data, such as filling with mean values.
- **Visualizations**: Confusion matrix, ROC curve, precision-recall curve, and more.

## :rocket: How to Run XplainIt on Your Local Machine

### Prerequisites

Ensure you have Python installed. This project requires Python 3.7 or above.

### Steps

1. Clone the repository:
   ```
   $ git clone https://github.com/turkerbdonmez/XplainIt.git
   $ cd XplainIt
   ```

2. Install the requirements:
   ```
   $ pip install -r requirements.txt
   ```

3. Run the app:
   ```
   $ streamlit run app.py
   ```

4. Open your browser and navigate to the provided local URL (e.g., `http://localhost:8501`).

## :hammer_and_wrench: Example Workflow

### 1. Load Data File
- Upload your data file (up to 200MB).
- Example: `dataset_final.csv` loaded with `,` as the delimiter and `utf-8` encoding.

### 2. Detect and Convert Categories (Optional)
- Automatically detect and convert categorical columns to numeric format.
- Example: No categorical columns detected in this workflow.

### 3. Specify Target & Features
- Target Column: `Outcome`
- Feature Columns: `Cholesterol`, `Hemoglobin`, `Insulin Resistance`, `Blood Glucose`, `Calcium`

### 4. Handle Missing Data and Basic Stats
- Missing Data Strategy: `Fill median`
- Train/Test Split Ratio (Comparison): `0.3`

### 5. Evaluate Models
- Compare default models and display evaluation metrics.

### 6. Final Model Selection & SHAP/LIME Analysis
- Final Model: `LightGBM`
- Train/Test Split Ratio (Final): `0.25`

#### SHAP Analysis
- Generate SHAP summaries, feature importance (bar), and dependence plots.
- Local SHAP plots generated with titles: `SHAP Plot Instance {id}`.

#### LIME Analysis
- Explanation format: `Explanation for Instance {id}`
- Plot type: `Both` (HTML and custom bar chart).
- Number of features displayed: `5`

## :file_folder: Directory Structure

```
XplainIt/
├── app.py                   # Main application file
├── shap_lime_utils.py       # SHAP and LIME utility functions
├── model_utils.py           # Model training and evaluation utilities
├── requirements.txt         # Python dependencies
└── output_images/           # Folder for generated plots (created at runtime)
```

## :tada: Contribute

Feel free to submit issues or pull requests to enhance the project. Contributions are welcome!

## :memo: License

This project is licensed under the MIT License. See the LICENSE file for details.
