# How to Run This Project

**No coding knowledge needed.** Follow these steps in order.

---

## What you will get

After running, you will have **one HTML file** that opens in any web browser.
It contains the full presentation: charts, tables, model results, reward
analysis and campaign simulation. You can print it, save it as PDF, or email it.

> **Important:** the dataset and the campaign-response labels are **synthetically
> generated for academic demonstration**, because real customer campaign data is
> unavailable. The results demonstrate the methodology — they are not real
> business outcomes.

---

## Before you start

You need **Python 3** installed on the computer.

**Check if you already have it:** press `Win + R`, type `cmd`, press Enter,
then type `python --version` and press Enter.

- If you see something like `Python 3.12.7` → you are ready, go to Step 1.
- If you see an error → install it first (see below).

### Installing Python (only if you do not have it)

1. Go to <https://www.python.org/downloads/>
2. Click the big yellow **"Download Python 3.x.x"** button
3. Run the downloaded file
4. **IMPORTANT:** on the first screen, tick the box that says
   **"Add python.exe to PATH"** (at the bottom of the installer window)
5. Click **"Install Now"** and wait for it to finish

---

## Step 1 — Get the project folder

You should have a folder called `department_store_reward_ml`.

**From a zip file:** right-click the zip file → **Extract All** → Extract.
Remember where you extracted it.

**From GitHub:** click the green **Code** button on the GitHub page →
**Download ZIP** → right-click → **Extract All**.

---

## Step 2 — Run it (one click)

Open the `department_store_reward_ml` folder.

**Double-click `run_everything.bat`**

That is it. A black window will open and start working.

### What is happening (just so you know)

| Step | What it does | Time |
|---|---|---|
| 1 | Checks Python is installed | instant |
| 2 | Installs any missing libraries | 1–2 min first time |
| 3 | Generates the dataset | a few seconds |
| 4 | Trains five machine-learning models | 30–60 seconds |
| 5 | Builds the presentation report | about 10 seconds |
| 6 | Opens the report in your browser | instant |

**Do not close the black window until it says "ALL DONE"** at the bottom.

---

## Step 3 — Your report opens automatically

When it finishes, your web browser opens showing the presentation report.

The report contains:

1. Executive summary with key numbers
2. System architecture diagram
3. Data overview and validation results
4. Customer segmentation (RFM and clustering)
5. Five-model machine-learning benchmark
6. What drives customer response
7. The reward engine and how offers are chosen
8. Example customer profile cards
9. Campaign priority list and simulation results
10. Limitations and recommendations

### To save it as PDF

In your browser, press **Ctrl + P** (or File → Print), then choose
**"Save as PDF"** as the destination, and click Save.

---

## Where the other files are

Everything is inside the `department_store_reward_ml` folder:

```
department_store_reward_ml/
│
├── run_everything.bat          ← the file you double-click
├── README.md                   ← full technical documentation
├── START_HERE.md               ← this file
│
├── data/                       ← the generated dataset (CSV files)
├── models/                     ← the trained model files
├── notebooks/                  ← the detailed Jupyter notebook
├── outputs/
│   ├── report/
│   │   └── project_report.html ← YOUR REPORT (open this)
│   ├── figures/                ← all the chart images
│   └── reports/                ← result tables (CSV)
│
└── src/                        ← the source code
```

---

## What if something goes wrong?

| Problem | What to do |
|---|---|
| "Python was not found" | Install Python (see above). Make sure you ticked **"Add Python to PATH"**. Then try again. |
| The window closes instantly | Right-click `run_everything.bat` → **Run as administrator**, or open `cmd`, `cd` to the folder, and type `run_everything.bat` so you can read the error. |
| "Some packages may be missing" | That is just a warning. The script tries again automatically. Wait for it to finish. |
| The report does not open | Open the folder `outputs\report` and double-click `project_report.html` yourself. |
| A library is missing | Open `cmd`, go to the project folder, and type: `pip install pandas numpy scikit-learn matplotlib xgboost catboost` |

---

## Want to see the detailed notebook instead?

The notebook (`notebooks/customer_reward_ml.ipynb`) shows every step in
detail — all the maths, every chart, and the reasoning behind each decision.

### Option A — Google Colab (easiest, nothing to install)

1. Open the project on GitHub
2. Open `notebooks/customer_reward_ml.ipynb`
3. At the top of the notebook, click the blue **"Open in Colab"** badge
   (or go to <colab.research.google.com> and open the notebook from GitHub)
4. In Colab, click **Runtime → Run all**

The notebook runs in Google's cloud — your computer only needs a browser.

### Option B — Jupyter on your computer

```powershell
cd department_store_reward_ml\notebooks
pip install notebook
jupyter notebook customer_reward_ml.ipynb
```

A browser window opens. Click **Kernel → Restart & Run All**.

---

## Presenting to others

The recommended way to present this project:

1. **Show the HTML report** (`outputs\report\project_report.html`) — this is
   the finished presentation
2. **Mention the methodology** — five models compared fairly, leak-free
   features, cost-aware reward engine, budget-matched simulation
3. **State the limitation clearly** — the data is synthetic; the value is in
   the pipeline and the decision logic, not in claimed real-world results

The report itself already contains this limitation statement, so it cannot be
presented out of context by accident.

---

## Quick reference

| What | Command |
|---|---|
| Run everything | Double-click `run_everything.bat` |
| Regenerate the data | `python src\data_generator.py` |
| Run the pipeline only | `python run_pipeline.py` |
| Build the report only | `python src\report_builder.py` |
| Run all checks | `python scripts\check_all.py` |
