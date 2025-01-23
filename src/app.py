from flask import Flask, request, render_template, redirect, url_for, flash
from src.plotting import prepare_data
from src.imu_extraction import extract_imu_data
import pandas as pd
import numpy as np
import json
from pathlib import Path
import os
from config import config
from dotenv import load_dotenv

load_dotenv('.env')

app = Flask(__name__, template_folder='../designer-interface')
app.config['ENV'] = os.getenv('FLASK_ENV')
app.config['DEBUG'] = os.getenv('FLASK_DEBUG') == '1'
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY')
app.config['FLASK_RUN_PORT'] = os.getenv('FLASK_RUN_PORT')

app.config['UPLOAD_FOLDER'] = config.DATA_DIR / 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1000 * 1024 * 1024 # 50gb limit

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Test df
df = None
settings = None

@app.route('/')
def upload_page():
    return render_template('welcome_tab.html')

@app.route('/predict_continue.html', methods=['GET'])
def predict_continue():
    return render_template('predict_continue.html', show_model_upload=True)

@app.route('/alternative_continue', methods=['GET'])
def predict_cont_no_show():
    return render_template('predict_continue.html', show_model_upload=False)

@app.route('/training', methods=['POST'])
def training():
    return render_template('training.html')

@app.route('/predict', methods=['POST'])
def predict():
    return render_template('predict.html')

@app.route('/download_model', methods=['GET'])
def download_model():
    # Empty route for you to implement the download functionality
    pass

@app.route('/data/uploads', methods=['POST'])
def handle_upload():
    global df
    global settings
    if request.method == 'POST':
        # Handle file uploads
        mp4_file = request.files.get('mp4-upload')
        csv_file = request.files.get('CSV-upload')
        # Get form data
        settings = {
            'a1': request.form.get('a1', 3),
            'a2': request.form.get('a2', 3),
            'a3': request.form.get('a3', 3),
            'a4': request.form.get('a4', 3),
            'a5': request.form.get('a5', 3)
        }
        
        # Save files and get their paths
        file_paths = {}
        
        if mp4_file and mp4_file.filename:
            app.config['UPLOAD_FOLDER']
            mp4_path = app.config['UPLOAD_FOLDER'] / mp4_file.filename
            mp4_file.save(mp4_path)
            file_paths['mp4_path'] = str(mp4_path)
            
            
        if csv_file and csv_file.filename:
            if not mp4_file and mp4_file.filename:
                flash('Error: MP4 file is required when uploading a CSV.')
                return redirect(url_for('upload_page'))
            csv_path = app.config['UPLOAD_FOLDER'] / csv_file.filename
            csv_file.save(csv_path)
            file_paths['csv_path'] = str(csv_path)

        if 'mp4_path' in file_paths and 'csv_path' in file_paths:
            try:
                df_t = pd.read_csv(file_paths['csv_path'])
            except Exception as e:
                print(f"Error while reading CSV: {e}")
                flash("An error occurred while reading the CSV file.")
                return redirect(url_for('upload_page'))
        
            required_columns = ["TIMESTAMP", "ACCL_x", "ACCL_y", "ACCL_z", "GYRO_x", "GYRO_y", "GYRO_z"]

            # Create a new DataFrame with only the required columns in the specified order
            df_t2 = pd.DataFrame({col: df_t[col] for col in required_columns if col in df_t.columns})

            # Check if any required columns are missing
            if list(df_t2.columns) != required_columns:
                flash("CSV file does not have the required columns (Capital sensitive): TIMESTAMP, ACCL_x, ACCL_y, ACCL_z, GYRO_x, GYRO_y, GYRO_z", "error")
                return redirect(url_for('upload_page'))
            df = df_t2
            return redirect(url_for('test'))

        # If 'mp4_path' is present, run the extract method
        if 'mp4_path' in file_paths and not 'csv_path' in file_paths:
            try:
                df = extract_imu_data(file_paths['mp4_path'])
                return redirect(url_for('test'))
            except Exception as e:
                print(f"Error while reading extracting IMU data: {e}")
                flash("An error occurred while trying to extract the IMU data.")
                return redirect(url_for('upload_page'))
            
        flash("You need to upload something")
        return redirect(url_for('upload_page'))


@app.route('/plot')
def plot():
    return Path('designer-interface/plot.html').read_text()

@app.route('/get_plot_data')
def get_plot_data():
    x_col = request.args.get('x', 'PC_1')
    y_col = request.args.get('y', 'PC_2')

    principal_df, mapping = prepare_data(df)

    data = {
        'x': principal_df[x_col].tolist(),
        'y': principal_df[y_col].tolist(),
        'frames': principal_df['FRAME'].tolist(),
        'colours': principal_df['COLOUR'].tolist(),
        'legend': mapping
    }
    return app.response_class(
        response=json.dumps(data),
        status=200,
        mimetype='application/json'
    )
