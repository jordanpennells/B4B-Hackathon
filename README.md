# Python-Marbling

## AI Marbling Grader Setup Instructions

Follow the steps below to get the AI Marbling Grader app running from scratch.

### 1. Set up your Python environment

1. Open PowerShell (or Terminal).
2. Navigate to your project folder, for example:
   ```powershell
   cd "C:\\Users\\pen253\\OneDrive - CSIRO\\2022\\Bytes for Bites Hackathon\\Python-Marbling"
   ```
3. Create and activate a virtual environment (if not already done):
   ```powershell
   python -m venv .venv
   .venv\Scripts\activate
   ```

### 2. Install dependencies

With the virtual environment activated, install the project dependencies:

```bash
pip install -r requirements.txt
```

If you do not have a `requirements.txt`, install the key packages manually:

```bash
pip install streamlit pandas numpy scikit-learn matplotlib pillow scikit-image joblib
```

### 3. Prepare your data

1. Ensure you have `features.csv` containing the extracted features (fat %, lacunarity, orientation, etc.).
2. Run the marbling index builder to generate the derived scores:
   ```bash
   python learn_mi.py --features features.csv
   ```

This command produces:
- `reports/mi_scores.csv` (handcrafted, LDA, logistic MI values)
- Violin plots and scatterplots for sanity checks
- `reports/mi_formulas.txt` (readable formulas and coefficients)

### 4. Train the marbling grader

Execute the training script:

```bash
python train_grader.py --features features.csv
```

This generates:
- `grader_model.pkl` (trained and calibrated model, features, and labels)
- `reports/feature_importance.csv` (feature importance values)
- `reports/confusion_matrix.png` (evaluation snapshot)

### 5. Run the dashboard app

Start Streamlit from your project root:

```bash
streamlit run app.py
```

Your app will open locally at a URL similar to:

```
http://localhost:8501
```
