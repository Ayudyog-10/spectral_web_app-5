import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from scipy.signal import savgol_filter
from fpdf import FPDF  # For PDF generation
import os
import logging
import time  # For timestamp in report
from pathlib import Path
from scipy.signal import savgol_filter  # For Savitzky-Golay
import numpy as np  # For SNV and MSC

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === Constants ===
PLOT_DIR = "spectral_web_app/static/plots"
os.makedirs(PLOT_DIR, exist_ok=True)

# === Preprocessing Functions ===
def snv(X):
    return (X - np.mean(X, axis=1, keepdims=True)) / np.std(X, axis=1, keepdims=True)

def msc(X):
    mean_spec = np.mean(X, axis=0)
    return np.array([(s - np.polyfit(mean_spec, s, 1)[1]) / np.polyfit(mean_spec, s, 1)[0] for s in X])

def savitzky_golay(X, window=11, poly_order=2):
    return savgol_filter(X, window_length=window, polyorder=poly_order, deriv=1, axis=1)

# === Feature Selection ===
def select_wavelengths_cars(X, y, wavelengths):
    np.random.seed(42)
    num_features = X.shape[1]
    idx = np.sort(np.random.choice(num_features, size=max(1, num_features // 2), replace=False))
    return X[:, idx], wavelengths[idx], idx

def select_wavelengths_random_frog(X, y, wavelengths):
    np.random.seed(42)
    scores = np.random.rand(X.shape[1])
    top_n = max(1, int(0.4 * X.shape[1]))
    top_idx = np.sort(np.argsort(scores)[-top_n:])
    return X[:, top_idx], wavelengths[top_idx], top_idx

# === Plot Generation ===
def generate_plots(X_raw, y, wavelengths, selected_indices, y_test, y_pred, model_name):
    """Generate all plots during pipeline execution and return paths"""
    plot_paths = {}
    
    # 1. Actual vs Predicted
    try:
        plt.figure(figsize=(8, 6))
        plt.scatter(y_test, y_pred, alpha=0.7, edgecolor='k', s=80)
        plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--', lw=2)
        plt.xlabel('Actual Values', fontsize=12)
        plt.ylabel('Predicted Values', fontsize=12)
        plt.title(f'Actual vs Predicted ({model_name})', fontsize=14)
        plt.grid(True, alpha=0.3)
        path = os.path.join(PLOT_DIR, "actual_vs_predicted.png")
        plt.savefig(path, bbox_inches='tight', dpi=150)
        plt.close()
        plot_paths['actual_vs_predicted'] = path
    except Exception as e:
        logger.error(f"Error generating actual vs predicted plot: {str(e)}")

    # 2. PCA Visualization
    try:
        plt.figure(figsize=(8, 6))
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_raw[:, selected_indices])
        plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.7, s=60)
        plt.colorbar(label='Target Value')
        plt.xlabel('Principal Component 1', fontsize=12)
        plt.ylabel('Principal Component 2', fontsize=12)
        plt.title('PCA of Selected Features', fontsize=14)
        plt.grid(True, alpha=0.3)
        path = os.path.join(PLOT_DIR, "pca_plot.png")
        plt.savefig(path, bbox_inches='tight', dpi=150)
        plt.close()
        plot_paths['pca_plot'] = path
    except Exception as e:
        logger.error(f"Error generating PCA plot: {str(e)}")

    # 3. Preprocessed Spectra Plots
    preprocessors = {
        "SNV": snv,
        "SG": savitzky_golay,
        "MSC": msc,
        "SNV+SG": lambda x: savitzky_golay(snv(x))
    }

    for name, func in preprocessors.items():
        try:
            plt.figure(figsize=(10, 6))
            X_proc = func(X_raw)
            for spectrum in X_proc[:100]:  # Plot first 100 spectra for clarity
                plt.plot(wavelengths, spectrum, alpha=0.3, lw=0.8)
            plt.title(f'{name} Processed Spectra', fontsize=14)
            plt.xlabel('Wavelength (nm)', fontsize=12)
            plt.ylabel('Absorbance', fontsize=12)
            plt.grid(True, alpha=0.2)
            path = os.path.join(PLOT_DIR, f"{name.lower()}_spectra.png")
            plt.savefig(path, bbox_inches='tight', dpi=150)
            plt.close()
            plot_paths[f"{name.lower()}_spectra"] = path
        except Exception as e:
            logger.error(f"Error generating {name} spectra plot: {str(e)}")

    return plot_paths

# === Model Evaluation ===
def evaluate_model(X, y, model):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    return rmse, r2, model, X_test, y_test, preds

def calculate_q2(X, y, model, cv=5):
    y_cv = cross_val_predict(model, X, y, cv=cv)
    return 1 - np.sum((y - y_cv)**2) / np.sum((y - np.mean(y))**2)

def find_optimal_pls_components(X, y, max_components=20):
    rmse_cv = []
    max_allowed = min(X.shape[0], X.shape[1], max_components)
    for n in range(1, max_allowed + 1):
        model = PLSRegression(n_components=n)
        y_cv = cross_val_predict(model, X, y, cv=5)
        rmse = np.sqrt(mean_squared_error(y, y_cv))
        rmse_cv.append(rmse)
    return np.argmin(rmse_cv) + 1, rmse_cv

# === Pipeline ===
def run_pipeline(data_path, wavelength_path):
    try:
        df = pd.read_excel(data_path, header=None)
        y = df.iloc[:, 0].values
        X_raw = df.iloc[:, 1:].values
        wave_values = pd.read_excel(wavelength_path, header=None).iloc[:, 0].values

        preprocessors = {
            "SNV": snv,
            "SG": savitzky_golay,
            "MSC": msc,
            "SNV+SG": lambda x: savitzky_golay(snv(x))
        }

        models = {
            "PLS": lambda n: PLSRegression(n_components=n),
            "RF": lambda _: RandomForestRegressor(n_estimators=100, random_state=42),
            "SVM": lambda _: SVR(),
            "ANN": lambda _: MLPRegressor(hidden_layer_sizes=(256, 128, 64), max_iter=2000, early_stopping=True, random_state=42)
        }

        feature_methods = {
            "CARS": select_wavelengths_cars,
            "RandomFrog": select_wavelengths_random_frog
        }

        results = []
        best_model_info = {"rmse": float("inf")}

        for feat_name, feat_func in feature_methods.items():
            X_sel, sel_waves, selected_indices = feat_func(X_raw, y, wave_values)
            for prep_name, prep_func in preprocessors.items():
                X_proc = prep_func(X_sel)
                for pca_state in [False, True]:
                    tag = "PCA" if pca_state else "Full"
                    X_final = PCA(n_components=0.95).fit_transform(X_proc) if pca_state else X_proc
                    for model_name, model_func in models.items():
                        best_n = find_optimal_pls_components(X_final, y)[0] if model_name == "PLS" else None
                        model = model_func(best_n)
                        rmse, r2, trained_model, X_test, y_test, y_pred = evaluate_model(X_final, y, model)
                        q2 = calculate_q2(X_final, y, model)
                        results.append({
                            "FeatureMethod": feat_name,
                            "Preprocessing": prep_name,
                            "Model": model_name,
                            "Approach": tag,
                            "LatentVariables": best_n if pca_state or model_name == "PLS" else "N/A",
                            "RMSE": rmse,
                            "R2": r2,
                            "Q2": q2
                        })
                        if rmse < best_model_info["rmse"]:
                            best_model_info = {
                                "name": f"{feat_name}_{prep_name}_{model_name}_{tag}",
                                "rmse": rmse,
                                "r2": r2,
                                "q2": q2,
                                "model": trained_model,
                                "wavelengths": sel_waves,
                                "selected_indices": selected_indices,
                                "X_test": X_test,
                                "y_test": y_test,
                                "y_pred": y_pred,
                                "latent_variables": best_n if pca_state or model_name == "PLS" else "N/A"
                            }



        # Generate all plots for the best model
        plot_paths = generate_plots(
            X_raw, y, wave_values,
            best_model_info["selected_indices"],
            best_model_info["y_test"],
            best_model_info["y_pred"],
            best_model_info["name"]
        )
        best_model_info["plot_paths"] = plot_paths

        pd.DataFrame(results).to_csv("model_results.csv", index=False)


        with open("best_model.pkl", "wb") as f:
            pickle.dump(best_model_info, f)

        logger.info("Pipeline completed successfully")
        return best_model_info

    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise

# === Report Generation ===
def generate_report():
    """Generate PDF report using pre-generated plots"""
    try:
        with open("best_model.pkl", "rb") as f:
            best_model_info = pickle.load(f)

        # Initialize PDF
        pdf = FPDF(orientation='P') # Corrected orientation
        pdf.set_auto_page_break(auto=True, margin=15)

        # Cover Page
        pdf.add_page()
        # === Set Background Image ===
        # Adjust the size (w=210, h=297 for A4) or center as needed
        pdf.image("spectral_web_app/static/maxresdefault (1).jpg", x=0, y=0, w=210, h=150)
        pdf.image("spectral_web_app/static/1750851633281-removebg-preview.png", x=60, y=145, w=90)
        pdf.set_y(20)  # space
        pdf.set_font("Arial", 'B', 24)
        pdf.cell(0, 50, "Spectral Analysis Report", ln=True, align='C')
        pdf.ln(175)
        pdf.set_font("Arial", '', 16)
        pdf.multi_cell(0, 10, f"Analysis performed on: {time.strftime('%Y-%m-%d %H:%M:%S')}", align='C')
        pdf.ln(5)
        pdf.set_font("Arial", 'I', 14)
        pdf.multi_cell(0, 10, "Generated by MetaspeQ Spectral Intelligence Engine", align='C')

        # Model Summary Page
        pdf.add_page()
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(0, 10, "Model Summary", ln=True)
        pdf.ln(10)

        pdf.set_font("Arial", '', 12)
        info_lines = [
            f"Best Model: {best_model_info['name']}",
            f"RMSE: {best_model_info['rmse']:.4f}",
            f"R²: {best_model_info['r2']:.4f}",
            f"Q²: {best_model_info['q2']:.4f}",
            f"Latent Variables: {best_model_info['latent_variables']}",
            f"Feature Selection: {best_model_info['name'].split('_')[0]}",
            f"Number of Selected Wavelengths: {len(best_model_info['wavelengths'])}"
        ]

        for line in info_lines:
            pdf.multi_cell(0, 10, line)
            pdf.ln(5)


                # Add wavelengths page
        pdf.add_page()
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(0, 10, "Selected Wavelengths", ln=True)
        pdf.ln(10)

        pdf.set_font("Arial", '', 10)
        selected_wavelengths = sorted(best_model_info['wavelengths'])
        col_width = 30
        cols = 5
        for i in range(0, len(selected_wavelengths), cols):
            for j in range(cols):
                if i + j < len(selected_wavelengths):
                    pdf.cell(col_width, 10, f"{selected_wavelengths[i+j]:.2f} nm", border=1)
            pdf.ln()

        # Add plots to PDF
        plot_descriptions = {
            "actual_vs_predicted": "Actual vs Predicted Values",
            "pca_plot": "PCA Visualization of Selected Features",
            "snv_spectra": "Standard Normal Variate (SNV) Processed Spectra",
            "sg_spectra": "Savitzky-Golay Processed Spectra",
            "msc_spectra": "Multiplicative Scatter Correction (MSC) Processed Spectra",
            "snv+sg_spectra": "SNV + Savitzky-Golay Processed Spectra"
        }

        for plot_key, description in plot_descriptions.items():
            path = best_model_info["plot_paths"].get(plot_key)
            if path and os.path.exists(path):
                pdf.add_page()
                pdf.set_font("Arial", 'B', 16)
                pdf.cell(0, 10, description, ln=True)
                pdf.ln(10)
                pdf.image(path, x=10, y=30, w=pdf.w-20)
                pdf.set_y(pdf.h-20)
                pdf.set_font("Arial", 'I', 10)
      




                # Add results table (with improved layout)
        if os.path.exists("model_results.csv"):
            pdf.add_page()
            pdf.set_font("Arial", 'B', 16)
            pdf.cell(0, 10, "Model Evaluation Results", ln=True)
            pdf.ln(10)

            results_df = pd.read_csv("model_results.csv")

            # Adjust column widths based on content
            col_widths = [25, 25, 20,30, 15, 15, 15, 15]
            headers = ["Feature", "Process", "Model","LatentVariables", "Method", "RMSE", "R²", "Q²"]

            # Header
            pdf.set_font("Arial", 'B', 8)
            for i, header in enumerate(headers):
                pdf.cell(col_widths[i], 10, header, border=1, align='C')
            pdf.ln()

            # Rows
            pdf.set_font("Arial", '', 7)
            for _, row in results_df.iterrows():
                values = [
                    row['FeatureMethod'][:10] + ('..' if len(row['FeatureMethod']) > 10 else ''),
                    row['Preprocessing'][:10] + ('..' if len(row['Preprocessing']) > 10 else ''),                   
                    row['Model'][:8],
                    str(int(row['LatentVariables'])) if pd.notnull(row['LatentVariables']) else "NA",
                    row['Approach'],
                    f"{row['RMSE']:.3f}",
                    f"{row['R2']:.3f}",
                    f"{row['Q2']:.3f}"
                ]

                for i, value in enumerate(values):
                    pdf.cell(col_widths[i], 10, value, border=1)
                pdf.ln()

        # Save PDF
        pdf.output("Spectral_Report.pdf")
        logger.info(f"PDF report generated with {pdf.page_no()} pages")

    except Exception as e:
        logger.error(f"Report generation failed: {str(e)}")
        raise