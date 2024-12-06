
# Sales Prediction App

This is a simple Flask web application that predicts sales based on advertising spend in three categories: TV, Radio, and Newspaper. The app allows users to input their advertising spend in these categories and get a predicted sales value as output.

## Project Structure
```
project/
│
├── app.py                    # Main Flask application
├── final_model.pkl            # Trained machine learning model (Pickle file)
├── col_names.pkl              # Column names for the input data (Pickle file)
├── requirements.txt           # List of dependencies
│
├── templates/
│   └── index.html             # HTML file for the web interface
│
├── static/
│   ├── css/
│   │   └── styles.css         # Styles for the web interface
│   ├── js/
│   │   └── script.js          # JavaScript for handling form submission and API requests
│
└── README.md                  # This file
```

## Setup and Installation

1. **Clone this repository** or **download** the project folder.

2. **Create a virtual environment** (optional, but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install the required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Flask app**:
   ```bash
   python api.py
   ```

5. **Access the application** in your web browser by navigating to:
   ```
   http://127.0.0.1:5000/
   ```

## Features

- **Input Fields**: Users can enter their advertising spend in three categories: TV, Radio, and Newspaper.
- **Prediction**: After clicking "Get Prediction", the app makes a prediction based on the input values.
- **Styled Output**: The predicted sales value is displayed in a styled container with a hover effect and color changes based on success or error.

## Dependencies

- Flask
- Pandas
- Joblib
- scikit-learn

To install the dependencies, run:
```bash
pip install -r requirements.txt
```

## Requirements

- Python 3.x
- A trained machine learning model (`final_model.pkl`) and column names (`col_names.pkl`) must be available.

## Notes

- The trained model (`final_model.pkl`) should be a scikit-learn model that can predict sales based on the input data format.
- The column names file (`col_names.pkl`) should contain the column names used during training.

## Troubleshooting

- If the app does not work, ensure that the `final_model.pkl` and `col_names.pkl` files are in the correct location.
- Ensure that Flask and all other dependencies are installed correctly.
- Check the browser's console for any JavaScript errors and refer to the Flask server logs for backend issues.

