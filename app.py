# app.py

import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

from model_utils import (
    train_model,
    plot_and_save_confusion_matrix,
    plot_and_save_roc_curve,
    plot_and_save_pr_curve
)
from shap_lime_utils import (
    convert_categorical_to_numeric,
    compute_shap_values,
    save_shap_summary_plots,
    save_shap_positive_negative_bar,
    save_shap_dependence_plots,
    save_shap_local_for_all_instances,
    compute_and_save_lime_for_all_instances
)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from interpret.glassbox import ExplainableBoostingClassifier

def detect_delimiter(uploaded_file):
    import csv
    uploaded_file.seek(0)
    sample = uploaded_file.read(1024).decode('utf-8', errors='ignore')
    uploaded_file.seek(0)
    possible_delimiters = [',',';','\t','|']
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=possible_delimiters)
        return dialect.delimiter
    except:
        return None

def read_csv_with_fallback(uploaded_file, delimiter):
    encodings_to_try = ["iso-8859-9","utf-8","latin-1","cp1252"]
    for enc in encodings_to_try:
        try:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, delimiter=delimiter, encoding=enc)
            return df, enc
        except:
            pass
    raise ValueError("Could not read file with any fallback encodings.")

def handle_missing_data(df, strategy="No action"):
    if strategy == "No action":
        return df
    elif strategy == "Drop rows":
        return df.dropna()
    else:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if strategy == "Fill mean":
            for col in numeric_cols:
                df[col] = df[col].fillna(df[col].mean())
        elif strategy == "Fill median":
            for col in numeric_cols:
                df[col] = df[col].fillna(df[col].median())
        elif strategy == "Fill mode":
            for col in numeric_cols:
                m = df[col].mode(dropna=True)
                if len(m) > 0:
                    df[col] = df[col].fillna(m[0])
        return df

def identify_categorical(df):
    cat_candidates = []
    for col in df.columns:
        if df[col].dtype not in [np.int64, np.float64]:
            unique_vals = df[col].dropna().unique()
            if len(unique_vals) < 10:
                cat_candidates.append(col)
    return cat_candidates

def main():
    st.title("XplainIt - Final with SHAP & LIME")

    # --- Session State placeholders ---
    if "df_original" not in st.session_state:
        st.session_state["df_original"] = None
    if "df_converted" not in st.session_state:
        st.session_state["df_converted"] = None
    if "df_handled" not in st.session_state:
        st.session_state["df_handled"] = None
    if "compare_results" not in st.session_state:
        st.session_state["compare_results"] = None

    if "final_model" not in st.session_state:
        st.session_state["final_model"] = None
    if "X_train_f" not in st.session_state:
        st.session_state["X_train_f"] = None
    if "X_test_f" not in st.session_state:
        st.session_state["X_test_f"] = None
    if "y_train_f" not in st.session_state:
        st.session_state["y_train_f"] = None
    if "y_test_f" not in st.session_state:
        st.session_state["y_test_f"] = None
    if "label_final" not in st.session_state:
        st.session_state["label_final"] = None
    if "model_choice" not in st.session_state:
        st.session_state["model_choice"] = None
    if "final_res" not in st.session_state:
        st.session_state["final_res"] = None

    if "shap_summary_path" not in st.session_state:
        st.session_state["shap_summary_path"] = None
    if "shap_bar_path" not in st.session_state:
        st.session_state["shap_bar_path"] = None
    if "shap_posneg_path" not in st.session_state:
        st.session_state["shap_posneg_path"] = None
    if "dependence_plots_saved" not in st.session_state:
        st.session_state["dependence_plots_saved"] = False
    if "explainer" not in st.session_state:
        st.session_state["explainer"] = None
    if "shap_values" not in st.session_state:
        st.session_state["shap_values"] = None
    if "local_shap_done" not in st.session_state:
        st.session_state["local_shap_done"] = False
    if "lime_done" not in st.session_state:
        st.session_state["lime_done"] = False

    # --- Folder path ---
    default_folder = os.path.join(os.getcwd(), "output_images")
    raw_output_folder = st.text_input("Folder path:", default_folder)

    # Normalize the input path
    normalized_folder = os.path.expanduser(raw_output_folder)
    normalized_folder = os.path.abspath(normalized_folder)

    if st.button("Check/Create Folder"):
        if not os.path.exists(normalized_folder):
            os.makedirs(normalized_folder, exist_ok=True)
        st.success(f"Folder is ready: {normalized_folder}")
        st.session_state["output_folder"] = normalized_folder  # Save folder path for later use

    # --- Step 1: Load CSV ---
    st.header("1. Load CSV")
    file = st.file_uploader("Upload CSV:", type=["csv"])
    if file is not None:
        delimiter = detect_delimiter(file)
        if delimiter is None:
            delimiter = st.selectbox("Choose delimiter:", [",",";","\t","|"])
        df_orig, used_enc = read_csv_with_fallback(file, delimiter)
        st.session_state["df_original"] = df_orig
        st.write(f"Read with {used_enc}")
        st.write("Data Preview:")
        st.dataframe(df_orig.head())

    # --- Step 2: Convert Potential Categorical ---
    if st.session_state["df_original"] is not None:
        st.header("2. Convert Potential Categorical (Optional)")
        cat_candidates = identify_categorical(st.session_state["df_original"])
        st.write("Potentially Categorical Columns:", cat_candidates)
        convert_cats = st.checkbox("Convert these columns to numeric automatically?")
        if convert_cats and cat_candidates:
            df_copy = st.session_state["df_original"].copy()
            df_conv, cat_legends = convert_categorical_to_numeric(df_copy, cat_candidates)
            st.session_state["df_converted"] = df_conv
            st.write("Legends for auto-converted columns:")
            st.json(cat_legends)
        else:
            st.session_state["df_converted"] = st.session_state["df_original"].copy()

        # --- Step 3: Select Target & Features ---
        st.header("3. Select Target & Feature Variables")
        all_cols = st.session_state["df_converted"].columns.tolist()
        target_col = st.selectbox("Target Column", all_cols)
        feature_cols = st.multiselect("Feature Columns", [c for c in all_cols if c != target_col])

        # --- Step 4: Missing Data + Basic Stats + T/T ratio ---
        st.header("4. Missing Data Handling + Basic Stats + T/T ratio for Compare")
        ratio_for_compare = st.selectbox("Select Train/Test ratio (Compare):", [0.2,0.25,0.3])
        strat = st.selectbox("Missing Data Strategy:", [
            "No action","Drop rows","Fill mean","Fill median","Fill mode"
        ])
        apply_data_button = st.button("Apply Missing Data Strategy")
        if apply_data_button:
            if target_col and feature_cols:
                df_temp = st.session_state["df_converted"][[target_col]+feature_cols].copy()
                df_temp = handle_missing_data(df_temp, strategy=strat)
                st.write(f"**Rows:** {len(df_temp)}")
                st.write(f"**Columns:** {df_temp.shape[1]}")
                st.write("**Statistics:**")
                st.write(df_temp.describe())
                st.session_state["df_handled"] = df_temp
                st.write("Data after missing-value handling:")
                st.dataframe(df_temp.head())
            else:
                st.error("Please select target & features first.")

        # --- Step 5: Compare Models ---
        st.header("5. Compare Models (Default)")
        compare_button = st.button("Compare Models Now")
        if compare_button:
            if st.session_state["df_handled"] is None:
                st.error("You must apply missing data handling first.")
            else:
                df_cmp = st.session_state["df_handled"].copy()
                df_cmp = df_cmp.dropna(subset=[target_col])
                df_cmp[target_col] = pd.to_numeric(df_cmp[target_col], errors='coerce')
                df_cmp = df_cmp.dropna(subset=[target_col])
                if df_cmp.empty:
                    st.error("No rows remain after cleaning. Stop.")
                else:
                    X_compare = df_cmp[feature_cols]
                    y_compare = df_cmp[target_col]
                    label_compare = LabelEncoder()
                    y_cmp_enc = label_compare.fit_transform(y_compare)

                    X_train_cmp, X_test_cmp, y_train_cmp, y_test_cmp = train_test_split(
                        X_compare, y_cmp_enc, test_size=ratio_for_compare, random_state=42
                    )
                    models_dict = {
                        "Logistic Regression": LogisticRegression(),
                        "Support Vector Machine": SVC(probability=True),
                        "Decision Tree": DecisionTreeClassifier(),
                        "Random Forest": RandomForestClassifier(),
                        "KNN": KNeighborsClassifier(),
                        "Naive Bayes": GaussianNB(),
                        "XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
                        "Adaboost": AdaBoostClassifier(),
                        "LightGBM": lgb.LGBMClassifier(),
                        "CatBoost": CatBoostClassifier(verbose=False),
                        "Explainable Boosting Machine": ExplainableBoostingClassifier(),
                        "Extra Trees": ExtraTreesClassifier()
                    }
                    results = []
                    for m_name, m_obj in models_dict.items():
                        try:
                            res = train_model(m_obj, X_train_cmp, y_train_cmp, X_test_cmp, y_test_cmp)
                            results.append({
                                "Model": m_name,
                                "Accuracy": res["accuracy"],
                                "F1":       res["f1"],
                                "Precision":res["precision"],
                                "Recall":   res["recall"],
                                "AUC-ROC":  res["roc_auc"],
                                "PR-AUC":   res["pr_auc"]
                            })
                        except Exception as e:
                            results.append({
                                "Model": m_name,
                                "Accuracy": None,
                                "F1":       None,
                                "Precision":None,
                                "Recall":   None,
                                "AUC-ROC":  None,
                                "PR-AUC":   None
                            })
                            st.error(f"{m_name} failed: {e}")
                    st.session_state["compare_results"] = results

        if st.session_state["compare_results"] is not None:
            st.write("Comparison Results (remains visible):")
            st.dataframe(st.session_state["compare_results"])

        # --- Step 6: Final Model & T/T ratio, Then SHAP & LIME ---
        st.header("6. Choose Model & T/T ratio, Then SHAP & LIME")
        model_choice = st.selectbox("Final Model Choice:", [
            "Logistic Regression","Support Vector Machine","Decision Tree","Random Forest","KNN",
            "Naive Bayes","XGBoost","Adaboost","LightGBM","CatBoost",
            "Explainable Boosting Machine","Extra Trees"
        ])
        ratio_for_final = st.selectbox("Train/Test ratio (Final):", [0.2,0.25,0.3])

        final_run_button = st.button("Train Selected Model")
        if final_run_button:
            if st.session_state["df_handled"] is None:
                st.error("You must handle data first.")
            else:
                df_final = st.session_state["df_handled"].copy()
                df_final = df_final.dropna(subset=[target_col])
                df_final[target_col] = pd.to_numeric(df_final[target_col], errors='coerce')
                df_final = df_final.dropna(subset=[target_col])
                if df_final.empty:
                    st.error("No valid rows remain. Stop.")
                else:
                    X_final = df_final[feature_cols]
                    y_final = df_final[target_col]
                    label_final = LabelEncoder()
                    y_final_enc = label_final.fit_transform(y_final)

                    chosen_map = {
                        "Logistic Regression": LogisticRegression(),
                        "Support Vector Machine": SVC(probability=True),
                        "Decision Tree": DecisionTreeClassifier(),
                        "Random Forest": RandomForestClassifier(),
                        "KNN": KNeighborsClassifier(),
                        "Naive Bayes": GaussianNB(),
                        "XGBoost": xgb.XGBClassifier(use_label_encoder=False,eval_metric='logloss'),
                        "Adaboost": AdaBoostClassifier(),
                        "LightGBM": lgb.LGBMClassifier(),
                        "CatBoost": CatBoostClassifier(verbose=False),
                        "Explainable Boosting Machine": ExplainableBoostingClassifier(),
                        "Extra Trees": ExtraTreesClassifier()
                    }
                    chosen_model = chosen_map[model_choice]

                    X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(
                        X_final, y_final_enc, test_size=ratio_for_final, random_state=42
                    )
                    X_train_f = X_train_f.reset_index(drop=True)
                    y_train_f = pd.Series(y_train_f).reset_index(drop=True)
                    X_test_f = X_test_f.reset_index(drop=True)
                    y_test_f = pd.Series(y_test_f).reset_index(drop=True)

                    final_res = train_model(chosen_model, X_train_f, y_train_f, X_test_f, y_test_f)

                    st.session_state["final_model"] = chosen_model
                    st.session_state["X_train_f"] = X_train_f
                    st.session_state["X_test_f"] = X_test_f
                    st.session_state["y_train_f"] = y_train_f
                    st.session_state["y_test_f"] = y_test_f
                    st.session_state["label_final"] = label_final
                    st.session_state["model_choice"] = model_choice
                    st.session_state["final_res"] = final_res

                    # Reset SHAP/LIME states
                    st.session_state["shap_summary_path"] = None
                    st.session_state["shap_bar_path"] = None
                    st.session_state["shap_posneg_path"] = None
                    st.session_state["dependence_plots_saved"] = False
                    st.session_state["explainer"] = None
                    st.session_state["shap_values"] = None
                    st.session_state["local_shap_done"] = False
                    st.session_state["lime_done"] = False

                    st.write("**Training Results**")
                    st.write("Accuracy:", final_res["accuracy"])
                    st.write("F1:", final_res["f1"])
                    st.write("Precision:", final_res["precision"])
                    st.write("Recall:", final_res["recall"])
                    st.write("AUC-ROC:", final_res["roc_auc"])
                    st.write("PR-AUC:", final_res["pr_auc"])

                    st.write("Confusion Matrix (Text):")
                    st.write(final_res["cm"])
                    
                    # Use output folder from session state
                    output_folder = st.session_state.get("output_folder", ".")
                    
                    cm_path = os.path.join(output_folder, f"{model_choice}_cm.png")
                    plot_and_save_confusion_matrix(final_res["cm"], cm_path, model_choice)
                    st.image(cm_path, caption="Confusion Matrix PNG")

                    if final_res["y_prob"] is not None:
                        roc_p = os.path.join(output_folder, f"{model_choice}_roc.png")
                        plot_and_save_roc_curve(st.session_state["y_test_f"], final_res["y_prob"], roc_p, model_choice)
                        st.image(roc_p, caption="ROC Curve PNG")

                        pr_p = os.path.join(output_folder, f"{model_choice}_pr.png")
                        plot_and_save_pr_curve(st.session_state["y_test_f"], final_res["y_prob"], pr_p, model_choice)
                        st.image(pr_p, caption="Precision-Recall Curve PNG")

    # --- SHAP & LIME Section ---
    if st.session_state["final_model"] is not None and st.session_state["final_res"] is not None:
        st.header("SHAP Analysis")
        st.markdown("### 1) Generate SHAP Summaries")
        shap_summary_button = st.button("Generate & Show SHAP Summaries")
        if shap_summary_button:
            try:
                chosen_model = st.session_state["final_model"]
                X_train_f = st.session_state["X_train_f"]
                X_test_f = st.session_state["X_test_f"]
                model_choice = st.session_state["model_choice"]

                # Use output folder from session state
                output_folder = st.session_state.get("output_folder", ".")
                
                explainer, shap_values = compute_shap_values(chosen_model, X_train_f, X_test_f, model_choice)
                save_shap_summary_plots(explainer, shap_values, X_test_f, output_folder, model_choice)
                save_shap_positive_negative_bar(chosen_model, X_test_f, output_folder, model_choice)

                st.session_state["explainer"] = explainer
                st.session_state["shap_values"] = shap_values

                st.session_state["shap_summary_path"] = os.path.join(output_folder, f"{model_choice}_shap_summary.png")
                st.session_state["shap_bar_path"] = os.path.join(output_folder, f"{model_choice}_shap_bar.png")
                st.session_state["shap_posneg_path"] = os.path.join(output_folder, f"{model_choice}_shap_pos_neg.png")

                save_shap_dependence_plots(explainer, shap_values, X_test_f, output_folder, model_choice)
                st.session_state["dependence_plots_saved"] = True

            except Exception as e:
                st.error(f"SHAP Summaries Error: {e}")

        if st.session_state["shap_summary_path"]:
            st.image(st.session_state["shap_summary_path"], caption="SHAP Summary")
        if st.session_state["shap_bar_path"]:
            st.image(st.session_state["shap_bar_path"], caption="Global SHAP Feature Importance (bar)")
        if st.session_state["shap_posneg_path"]:
            st.image(st.session_state["shap_posneg_path"], caption="Global SHAP Positive/Negative")
        if st.session_state["dependence_plots_saved"]:
            st.success("SHAP dependence plots saved.")

        st.markdown("### 2) Local SHAP Plots")
        user_title_format = st.text_input("Local SHAP title format (use {id}):", "Local SHAP idx={id}")
        shap_local_button = st.button("Generate Local SHAP for All Instances")
        if shap_local_button:
            if st.session_state["explainer"] is None or st.session_state["shap_values"] is None:
                st.error("Please generate the SHAP summaries first.")
            else:
                try:
                    chosen_model = st.session_state["final_model"]
                    X_train_f = st.session_state["X_train_f"]
                    X_test_f = st.session_state["X_test_f"]
                    y_train_f = st.session_state["y_train_f"]
                    y_test_f = st.session_state["y_test_f"]
                    label_final = st.session_state["label_final"]
                    model_choice = st.session_state["model_choice"]
                    explainer = st.session_state["explainer"]
                    shap_values = st.session_state["shap_values"]

                    # Use output folder from session state
                    output_folder = st.session_state.get("output_folder", ".")
                    
                    save_shap_local_for_all_instances(
                        chosen_model, explainer, shap_values,
                        X_train_f, X_test_f, y_train_f, y_test_f,
                        label_final, output_folder, model_choice,
                        user_title_format=user_title_format
                    )
                    st.session_state["local_shap_done"] = True
                except Exception as e:
                    st.error(f"Local SHAP Error: {e}")

        if st.session_state["local_shap_done"]:
            st.success("Local SHAP done. Check output folder for PNGs.")

        # --- LIME Analysis ---
        st.header("LIME Analysis")
        st.markdown("### 1) Provide Title Format, Plot Type & Number of Features")

        lime_title_format = st.text_input("LIME Title format (use {id}):", "Local Explanation for instance {id}")
        lime_plot_choice = st.selectbox("LIME Plot Type:", ["Original","Edited","Both"])
        num_features = st.number_input("Number of LIME features to display:", min_value=1, max_value=20, value=6)

        st.write(
            "Original -> saves the explanation in **HTML** (the standard LIME output) "
            "Edited -> the custom matplotlib red/green bar chart, "
            "Both -> do both."
        )

        lime_run_button = st.button("Run LIME for All Instances")
        if lime_run_button:
            try:
                chosen_model = st.session_state["final_model"]
                X_train_f = st.session_state["X_train_f"]
                X_test_f = st.session_state["X_test_f"]
                y_train_f = st.session_state["y_train_f"]
                y_test_f = st.session_state["y_test_f"]
                label_final = st.session_state["label_final"]
                model_choice = st.session_state["model_choice"]

                # Use output folder from session state
                output_folder = st.session_state.get("output_folder", ".")
                
                compute_and_save_lime_for_all_instances(
                    chosen_model, X_train_f, X_test_f,
                    y_train_f, y_test_f, label_final, output_folder, model_choice,
                    lime_plot_choice=lime_plot_choice,
                    lime_title_format=lime_title_format,
                    num_features=num_features
                )
                st.session_state["lime_done"] = True
            except Exception as e:
                st.error(f"LIME error: {e}")

        if st.session_state["lime_done"]:
            st.success("LIME done! Check the output folder. The 'Original' explanation is saved as HTML, and the 'Edited' explanation is the red/green PNG.")

if __name__=="__main__":
    main()
