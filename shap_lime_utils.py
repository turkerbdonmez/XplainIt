# shap_lime_utils.py

import os
import shap
import lime
import lime.lime_tabular
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
from PIL import Image
import streamlit as st
from sklearn.preprocessing import LabelEncoder

def convert_categorical_to_numeric(df, cat_cols):
    legends = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = df[col].astype(str)
        df[col] = le.fit_transform(df[col])
        legend_map = {encoded: original for encoded, original in enumerate(le.classes_)}
        legends[col] = legend_map
    return df, legends

def compute_shap_values(model, X_train, X_test, model_name=None):
    tree_based = [
        "RandomForestClassifier","XGBClassifier","LGBMClassifier",
        "CatBoostClassifier","ExtraTreesClassifier","DecisionTreeClassifier"
    ]
    cname = type(model).__name__
    if any(t in cname for t in tree_based):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
    else:
        # For non-tree models, use KernelExplainer
        # (Note: model.predict_proba required for classification; adapt as needed.)
        smp = X_train.sample(min(50,len(X_train)), random_state=42)
        explainer = shap.KernelExplainer(model.predict_proba, smp)
        shap_values = explainer.shap_values(X_test)
    return explainer, shap_values

def save_shap_summary_plots(explainer, shap_values, X_test, output_folder, model_name="model"):
    plt.figure()
    shap.summary_plot(shap_values, X_test, feature_names=X_test.columns, show=False)
    p1 = os.path.join(output_folder, f"{model_name}_shap_summary.png")
    plt.savefig(p1, dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure()
    shap.summary_plot(shap_values, X_test, feature_names=X_test.columns, plot_type="bar", show=False)
    p2 = os.path.join(output_folder, f"{model_name}_shap_bar.png")
    plt.savefig(p2, dpi=300, bbox_inches='tight')
    plt.close()

def save_shap_positive_negative_bar(model, X_test, output_folder, model_name="model"):
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_test)
    if isinstance(shap_vals, list) and len(shap_vals) > 1:
        shap_vals = shap_vals[1]

    mean_pos, mean_neg = [], []
    for i in range(shap_vals.shape[1]):
        col_vals = shap_vals[:, i]
        pos_vals = col_vals[col_vals>0]
        neg_vals = col_vals[col_vals<0]
        mean_pos.append(np.mean(pos_vals) if len(pos_vals)>0 else 0)
        mean_neg.append(np.mean(neg_vals) if len(neg_vals)>0 else 0)
    mean_pos = np.array(mean_pos)
    mean_neg = np.array(mean_neg)
    mean_abs = np.abs(mean_pos + mean_neg)
    idx_sorted = np.argsort(mean_abs)[::-1]
    sorted_feats = X_test.columns[idx_sorted]
    pos_sorted = mean_pos[idx_sorted]
    neg_sorted = mean_neg[idx_sorted]

    plt.figure(figsize=(10,8))
    plt.barh(range(len(sorted_feats)), pos_sorted[::-1], color='red', label='Positive')
    plt.barh(range(len(sorted_feats)), neg_sorted[::-1], color='blue', label='Negative')
    plt.yticks(range(len(sorted_feats)), sorted_feats[::-1])
    plt.xlabel('Mean SHAP Value')
    plt.ylabel('Features')
    plt.title(f'Global SHAP Positive/Negative: {model_name}')
    plt.legend()
    save_pn = os.path.join(output_folder, f"{model_name}_shap_pos_neg.png")
    plt.savefig(save_pn, dpi=300, bbox_inches='tight')
    plt.close()

def save_shap_dependence_plots(explainer, shap_values, X_test, output_folder, model_name="model"):
    if isinstance(shap_values, list):
        shap_use = shap_values[1] if len(shap_values)>1 else shap_values[0]
    else:
        shap_use = shap_values

    dep_folder = os.path.join(output_folder, f"{model_name}_dependence_plots")
    os.makedirs(dep_folder, exist_ok=True)
    for feat in X_test.columns:
        buf = io.BytesIO()
        plt.figure()
        shap.dependence_plot(feat, shap_use, X_test, interaction_index=None, show=False)
        plt.savefig(buf, format='jpg', dpi=300, bbox_inches='tight')
        buf.seek(0)
        img = Image.open(buf)
        path_ = os.path.join(dep_folder, f"{model_name}_dep_{feat}.jpg")
        img.save(path_)
        plt.close()
        buf.close()

def save_shap_local_for_all_instances(
    model, explainer, shap_values,
    X_train, X_test, y_train, y_test,
    label_enc, out_dir, model_name="model",
    user_title_format="Local SHAP idx={id}"
):
    base_dir = os.path.join(out_dir, f"{model_name}_locals")
    for sub in ["TP","TN","FP","FN"]:
        os.makedirs(os.path.join(base_dir, sub), exist_ok=True)

    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)

    if isinstance(shap_values, list):
        shap_arr = shap_values[1] if len(shap_values)>1 else shap_values[0]
    else:
        shap_arr = shap_values

    progress_train = st.progress(0)
    progress_test = st.progress(0)

    def local_plot(i, is_train):
        if is_train:
            act_enc = y_train[i]
            pred_enc = train_preds[i]
            xrow = X_train.iloc[i]
            ds = "Train"
        else:
            act_enc = y_test[i]
            pred_enc = test_preds[i]
            xrow = X_test.iloc[i]
            ds = "Test"

        act_val = label_enc.inverse_transform([act_enc])[0]
        pred_val = label_enc.inverse_transform([pred_enc])[0]
        if act_enc==1 and pred_enc==1:
            outcome="TP"
        elif act_enc==1 and pred_enc==0:
            outcome="FN"
        elif act_enc==0 and pred_enc==0:
            outcome="TN"
        else:
            outcome="FP"

        loc_sub = os.path.join(base_dir,outcome)
        os.makedirs(loc_sub, exist_ok=True)

        if i >= shap_arr.shape[0]:
            return
        loc_sv = shap_arr[i]
        df_ = pd.DataFrame({"Feature":xrow.index,"SHAP":loc_sv})
        df_["abs"] = df_["SHAP"].abs()
        df_ = df_.sort_values("abs", ascending=True)

        instance_title = user_title_format.replace("{id}", str(i))

        plt.figure(figsize=(10,6))
        plt.barh(
            df_["Feature"],
            df_["SHAP"],
            color=df_["SHAP"].apply(lambda v: "red" if v>0 else "blue")
        )
        plt.title(instance_title)
        for idx_b, rowval in enumerate(df_["SHAP"]):
            plt.text(rowval, idx_b, f"{rowval:.4f}", va="center")
        fpath = os.path.join(loc_sub, f"{ds}_{i}_shap.png")
        plt.savefig(fpath,dpi=300,bbox_inches='tight')
        plt.close()

        txtp = os.path.join(loc_sub, f"{ds}_{i}_info.txt")
        with open(txtp,"w",encoding="utf-8") as fp:
            fp.write(f"Index: {i}\nDataset: {ds}\n")
            fp.write(f"Actual Enc: {act_enc} => {act_val}\n")
            fp.write(f"Pred Enc: {pred_enc} => {pred_val}\n")
            fp.write(f"Outcome: {outcome}\n\n")
            fp.write("X data:\n")
            fp.write(str(xrow))
            fp.write("\n\nSHAP details:\n")
            fp.write(str(df_))

    for i in range(len(X_train)):
        local_plot(i, True)
        progress_train.progress(int((i+1)*100/len(X_train)))

    for i in range(len(X_test)):
        local_plot(i, False)
        progress_test.progress(int((i+1)*100/len(X_test)))

def compute_and_save_lime_for_all_instances(
    model, X_train, X_test,
    y_train, y_test, label_enc,
    out_dir, model_name="model",
    lime_plot_choice="Original",
    lime_title_format="Local Explanation for instance {id}",
    num_features=6
):
    base_dir = os.path.join(out_dir, f"{model_name}_locals")
    for sub in ["TP","TN","FP","FN","REG"]:
        os.makedirs(os.path.join(base_dir, sub), exist_ok=True)

    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)

    progress_train = st.progress(0)
    progress_test = st.progress(0)

    # Classification vs. Regression
    is_classification = hasattr(model, "predict_proba")
    mode_type = "classification" if is_classification else "regression"
    predict_fn = model.predict_proba if is_classification else model.predict
    class_names = [str(c) for c in label_enc.classes_] if is_classification else None

    lime_explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=X_train.columns.tolist(),
        class_names=class_names,
        verbose=True,
        mode=mode_type
    )

    def do_lime(i, is_train):
        if is_train:
            aenc = y_train[i]
            penc = train_preds[i]
            xrow = X_train.iloc[i]
            ds="Train"
        else:
            aenc = y_test[i]
            penc = test_preds[i]
            xrow = X_test.iloc[i]
            ds="Test"

        if is_classification:
            aval = label_enc.inverse_transform([aenc])[0]
            pval = label_enc.inverse_transform([penc])[0]
            if aenc==1 and penc==1:
                outc="TP"
            elif aenc==1 and penc==0:
                outc="FN"
            elif aenc==0 and penc==0:
                outc="TN"
            else:
                outc="FP"
        else:
            aval = aenc
            pval = penc
            outc="REG"

        subp = os.path.join(base_dir, outc)
        os.makedirs(subp, exist_ok=True)

        title_str = lime_title_format.replace("{id}", str(i))

        try:
            exp = lime_explainer.explain_instance(
                data_row=xrow.values,
                predict_fn=predict_fn,
                num_features=num_features
            )

            # Original -> HTML
            if lime_plot_choice in ["Original","Both"]:
                html_content = exp.as_html()
                html_path = os.path.join(subp, f"{ds}_{i}_lime_original.html")
                with open(html_path, "w", encoding="utf-8") as f:
                    f.write(html_content)

            # Edited -> red/green bar chart
            if lime_plot_choice in ["Edited","Both"]:
                vals = exp.as_list()
                feat_names = [v[0] for v in vals]
                feat_scores = [v[1] for v in vals]
                fig_edit = plt.figure(figsize=(8,6))
                plt.title(title_str)
                colors = ["green" if sc>0 else "red" for sc in feat_scores]
                y_pos = np.arange(len(feat_names))
                plt.barh(y_pos, feat_scores, color=colors)
                plt.yticks(y_pos, feat_names)
                for idxv,barval in enumerate(feat_scores):
                    plt.text(barval, idxv, f"{barval:.4f}")
                sp_edit = os.path.join(subp, f"{ds}_{i}_lime_edited.png")
                fig_edit.savefig(sp_edit, dpi=300, bbox_inches='tight')
                plt.close(fig_edit)

            # Info text
            tpath = os.path.join(subp, f"{ds}_{i}_lime.txt")
            with open(tpath,"w",encoding="utf-8") as fp:
                fp.write(f"Index: {i}\nDataset: {ds}\n")
                fp.write(f"Actual: {aval}\n")
                fp.write(f"Pred: {pval}\n")
                fp.write(f"Outcome: {outc}\n\n")
                fp.write("X row:\n")
                fp.write(str(xrow))
                fp.write("\n\nLIME Explanation:\n")
                fp.write(str(exp.as_list()))
        except:
            pass

    for i in range(len(X_train)):
        do_lime(i, True)
        progress_train.progress(int((i+1)*100/len(X_train)))

    for i in range(len(X_test)):
        do_lime(i, False)
        progress_test.progress(int((i+1)*100/len(X_test)))
