from flask import Flask, request, render_template, send_file, redirect, url_for, flash, session
import os
from werkzeug.utils import secure_filename
from pipeline import run_pipeline, generate_report
import time
import logging


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)


app.secret_key = os.getenv('FLASK_SECRET_KEY', 'your_very_strong_secret_key_here')
app.config['UPLOAD_FOLDER'] = os.path.abspath('spectral_web_app/uploads')
app.config['ALLOWED_EXTENSIONS'] = {'xlsx', 'xls', 'csv'}  # Added 'csv'
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5MB limit
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        session.pop('_flashes', None)
    
    try:
        model_exists = os.path.exists(os.path.abspath("best_model.pkl"))
        report_exists = os.path.exists(os.path.abspath("Spectral_Report.pdf"))
        data_uploaded = 'data_file' in session
        wavelength_uploaded = 'wavelength_file' in session
        
        return render_template('index.html', 
                            model_exists=model_exists,
                            report_exists=report_exists,
                            data_uploaded=data_uploaded,
                            wavelength_uploaded=wavelength_uploaded)
    except Exception as e:
        logger.error(f"Error in index route: {str(e)}")
        flash("An error occurred while loading the page", "error")
        return render_template('index.html')

@app.route('/upload_data', methods=['POST'])
def upload_data():
    if 'data_file' not in request.files:
        flash('No file part', 'error')
        return redirect(url_for('index'))
    
    file = request.files['data_file']
    if file.filename == '':
        flash('No selected file', 'error')
        return redirect(url_for('index'))
    
    try:
        if file and allowed_file(file.filename):
            filename = secure_filename("uploaded_data.xlsx")
            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(path)
            session['data_file'] = filename
            flash("Spectral data uploaded successfully.", "success")
        else:
            flash("Invalid file type. Please upload an Excel file (.xlsx or .xls).", "error")
    except Exception as e:
        logger.error(f"Error uploading data file: {str(e)}")
        flash("Error uploading file", "error")
    
    return redirect(url_for('index'))

@app.route('/upload_wavelength', methods=['POST'])
def upload_wavelength():
    if 'wavelength_file' not in request.files:
        flash('No file part', 'error')
        return redirect(url_for('index'))
    
    file = request.files['wavelength_file']
    if file.filename == '':
        flash('No selected file', 'error')
        return redirect(url_for('index'))
    
    try:
        if file and allowed_file(file.filename):
            filename = secure_filename("uploaded_wavelength.xlsx")
            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(path)
            session['wavelength_file'] = filename
            flash("Wavelength file uploaded successfully.", "success")
        else:
            flash("Invalid file type. Please upload an Excel file (.xlsx or .xls).", "error")
    except Exception as e:
        logger.error(f"Error uploading wavelength file: {str(e)}")
        flash("Error uploading file", "error")
    
    return redirect(url_for('index'))

@app.route('/run_analysis', methods=['POST'])
def run_analysis():
    if not all(key in session for key in ['data_file', 'wavelength_file']):
        flash("Please upload both data and wavelength files first.", "error")
        return redirect(url_for('index'))
    
    data_path = os.path.join(app.config['UPLOAD_FOLDER'], session['data_file'])
    wave_path = os.path.join(app.config['UPLOAD_FOLDER'], session['wavelength_file'])

    try:
        start_time = time.time()
        run_pipeline(data_path, wave_path)
        elapsed = time.time() - start_time
        flash(f"Analysis completed successfully in {elapsed:.2f} seconds.", "success")
    except Exception as e:
        logger.error(f"Analysis error: {str(e)}")
        flash(f"Error during analysis: {str(e)}", "error")
    
    return redirect(url_for('index'))

@app.route('/generate_report', methods=['POST'])
def generate_report_route():
    if not os.path.exists("best_model.pkl"):
        flash("No analysis results found. Please run analysis first.", "error")
        return redirect(url_for('index'))
    
    try:
        start_time = time.time()
        generate_report()
        elapsed = time.time() - start_time
        flash(f"Report generated successfully in {elapsed:.2f} seconds.", "success")
    except Exception as e:
        logger.error(f"Report generation error: {str(e)}")
        flash(f"Error during report generation: {str(e)}", "error")
    
    return redirect(url_for('index'))

@app.route('/download_model', methods=['GET'])
def download_model():
    if not os.path.exists("best_model.pkl"):
        flash("No model found. Please run analysis first.", "error")
        return redirect(url_for('index'))
    
    try:
        return send_file(
            os.path.abspath("best_model.pkl"),
            as_attachment=True,
            download_name="spectral_model.pkl"
        )
    except Exception as e:
        logger.error(f"Error downloading model: {str(e)}")
        flash(f"Error downloading model: {str(e)}", "error")
        return redirect(url_for('index'))

@app.route('/download_report', methods=['GET'])
def download_report():
    if not os.path.exists("Spectral_Report.pdf"):
        flash("No report found. Please generate the report first.", "error")
        return redirect(url_for('index'))
    
    try:
        return send_file(
            os.path.abspath("Spectral_Report.pdf"),
            as_attachment=True,
            download_name="spectral_analysis_report.pdf"
        )
    except Exception as e:
        logger.error(f"Error downloading report: {str(e)}")
        flash(f"Error downloading report: {str(e)}", "error")
        return redirect(url_for('index'))

if __name__ == "__main__":
    app.run(debug=True, threaded=True)