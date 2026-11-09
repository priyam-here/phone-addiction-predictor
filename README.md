# Phone Addiction Level Predictor

Predicts a user's phone addiction level using demographic, psychological, and smartphone-usage behavioral data. We built a Linear Regression pipeline (using `pandas`, `scikit-learn`, `matplotlib`, and `seaborn`) and utilized Data Analysis to identify and filter out artificially capped synthetic data.

## Team and Roles

| Member | Entry no. | Role | 
| ----- | ----- | ----- | 
| Priyam Vatsa | 2026AIB1070 | Documentation lead | 
| Kukadiya Dhruvil Rajeshbhai | 2026MCB1537 | PPT lead | 
| Aniruddh Abhishek | 2026AIB1014 | Execution lead | 
| Nenavath Poojitha | 2026EEB1384 | Video lead | 

*All four members contributed functional Python code to the machine learning pipeline (see commit history).*

## Project Structure

```text
phone-addiction-predictor/
├── data/
│   └── phone_addiction_dataset.csv  # 6,000 rows, anonymised survey data
├── src/
│   ├── data_prep.py                 # Handles loading, cleaning, capping, and scaling
│   ├── train.py                     # Handles model training and metric evaluation
│   ├── plots.py                     # Generates scatter plots, heatmaps, and histograms
│   └── main.py                      # Main entry point running the 2-phase pipeline
├── outputs/                         # Generated plots are saved here automatically
│   ├── 0_correlation_heatmap.png                 
│   ├── 1_actual_vs_predicted.png                     
│   ├── 2_feature_importance.png                    
│   └── 3_residual_distribution.png 
│   └── 4_data_discovery.png 
├── requirements.txt                 # Pinned library versions
└── README.md
```

## Setup & Installation

Tested on Python 3.10+ (Windows, macOS, Ubuntu).

**1. Clone the repository**
```bash
git clone https://github.com/priyam-here/phone-addiction-predictor.git
cd phone-addiction-predictor
```

**2. Create a virtual environment**
```bash
python -m venv venv
```

**3. Activate the virtual environment**
* **On Windows:**
```bash
venv\Scripts\activate
```
* **On macOS/Linux:**
```bash
source venv/bin/activate
```
**4. Install Dependencies**
```bash
pip install -r requirements.txt
```
## How to Run

1. Run the main script from the root directory:
   ```bash
   python src/main.py
   ```
2. The script processes the data, prints the Phase 1 Overfitting demonstration, followed by the Phase 2 Clean Model metrics (Train vs. Test) to the terminal.
3. It will automatically save 5 evaluation visualizations in the newly created `outputs/` folder.
4. Expected runtime: < 15-20 seconds.

## Expected Output (Abridged)

```text
Initializing Phone Addiction Predictor Pipeline...
-> Initial raw dataset loaded: 6000 rows.

--- CORE FEATURE STATISTICS ---
1. Summary Statistics (Mean, Std, Min, Max):
       Daily_Usage_Hours  Sleep_Hours  Addiction_Level
count            6000.00      6000.00          6000.00
mean                5.02         6.49             8.88
std                 1.96         1.49             1.61
min                 0.00         3.00             1.00
25%                 3.70         5.50             8.00
50%                 5.00         6.50            10.00
75%                 6.40         7.50            10.00
max                11.50        10.00            10.00

2. Medians:
Daily_Usage_Hours     5.0
Sleep_Hours           6.5
Addiction_Level      10.0
dtype: float64

3. Covariance Matrix:
                   Daily_Usage_Hours  Sleep_Hours  Addiction_Level
Daily_Usage_Hours               3.83         0.05             1.89
Sleep_Hours                     0.05         2.22            -0.52
Addiction_Level                 1.89        -0.52             2.59
-------------------------------

-> Artificial caps removed. Final training rows: 2361, Testing rows: 591.
-> Generating Post-Processing EDA Correlation Heatmap...

--- CORE FEATURE STATISTICS AFTER CAP REMOVAL ---
1. Summary Statistics (Mean, Std, Min, Max):
       Daily_Usage_Hours  Sleep_Hours  Addiction_Level
count            2952.00      2952.00          2952.00
mean                3.88         6.78             7.73
std                 1.59         1.46             1.63
min                 0.00         3.00             1.00
25%                 2.90         5.80             6.70
50%                 3.90         6.80             8.00
75%                 4.90         7.80             9.00
max                 8.40        10.00             9.90

2. Medians:
Daily_Usage_Hours    3.9
Sleep_Hours          6.8
Addiction_Level      8.0
dtype: float64

3. Covariance Matrix:
                   Daily_Usage_Hours  Sleep_Hours  Addiction_Level
Daily_Usage_Hours               2.54         0.41             1.25
Sleep_Hours                     0.41         2.14            -0.39
Addiction_Level                 1.25        -0.39             2.64
-------------------------------

Training Linear Regression model...

--- MODEL RESULTS (TRAIN vs TEST) ---
R-squared (R2)  | Train: 0.9997 | Test: 0.9997
Mean Sq. Error  | Train: 0.0008 | Test: 0.0009
Mean Abs. Error | Train: 0.0241 | Test: 0.0252
-------------------------------------
Conclusion: Train and Test metrics are nearly identical, proving NO OVERFITTING.

Generating visualizations in 'outputs/' directory...
Pipeline complete.
```

## Links & Deliverables

* **Presentation Slides (5):**
* **Project Demo Video (max 5:00):**

## AI-Use Disclosure

* **Gemini:** Gemini: Diagnosed the dimensionality error caused by the Location column, generated parts of the code of data_prep.py and plots.py, and structured the README. All code was reviewed and tested locally.
Generated the base version of the PPT which was then heavily modified by us according to our needs and standards.

* All generated code was manually reviewed, split into modular scripts for team contribution, tested locally in our virtual environments, and the underlying logic is explained in our presentation video.
