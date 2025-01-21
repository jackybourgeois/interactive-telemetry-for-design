from flask import Flask, jsonify, request
from plotting import prepare_data
import pandas as pd
import numpy as np
import json

app = Flask(__name__)

# Custom JSON encoder to handle NaN values
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj) if not np.isnan(obj) else None
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

# Load and prepare data once when starting the server
df = pd.read_csv('C:\projects\interactive-telemetry-for-design\data\CSVs\GoPro_test.csv')  # Replace with your data file path
principal_df, mapping = prepare_data(df)

@app.route('/')
def index():
    return open('designer-interface/plot.html').read()

@app.route('/get_plot_data')
def get_plot_data():
    x_col = request.args.get('x', 'PC_1')
    y_col = request.args.get('y', 'PC_2')

    # Prepare the data for the plot
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

if __name__ == '__main__':
    app.run(debug=True)