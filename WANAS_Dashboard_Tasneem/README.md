# WANAS Dashboard

## Project Description

WANAS is a smart care system designed to support people with Alzheimer's disease. This dashboard allows caregivers to monitor patients, track medications, manage alerts, and review patient activity.

## 1. Requirements

Before running the project, make sure you have:

- Python 3.10 or newer
- Visual Studio Code (recommended)
- Internet connection for installing the required Python libraries

## 2. Project Structure

Make sure the project folder contains the following files and folders:

```text
WANAS_Dashboard/
│
├── app.py
├── requirements.txt
│
└── data/
    ├── sample_data.json
    └── alerts.json
```

The `data` folder stores patient information and saved alert history.

## 3. How to Install and Run the Project

### Step 1: Download the Project

Download or extract the WANAS Dashboard project folder to your computer.

Open the folder in Visual Studio Code.

### Step 2: Open the Terminal

In Visual Studio Code:

1. Click Terminal.
2. Select New Terminal.
3. Make sure the terminal is located inside the WANAS_Dashboard folder.

### Step 3: Create a Virtual Environment

Run:

```powershell
python -m venv venv
```

This creates an isolated Python environment for the project.

### Step 4: Activate the Virtual Environment

For Windows PowerShell, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
```

After activation, you should see `(venv)` at the beginning of your terminal line.

### Step 5: Install Required Libraries

Run:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

This installs all the libraries needed by the dashboard.

### Step 6: Run the Dashboard

Run:

```powershell
python -m streamlit run app.py
```

Streamlit will start the application and provide a local address, usually:

```text
http://localhost:8501
```

Open this address in your web browser if it does not open automatically.

## 4. How to Stop the Dashboard

To stop the application, return to the VS Code terminal and press:

```text
Ctrl + C
```

## 5. Important Notes

- Always run the dashboard from the main project folder.
- Do not remove the `data` folder or its JSON files.
- Patient information and medication updates are saved in `sample_data.json`.
- Alert history is saved in `alerts.json`.
- Do not delete or replace the JSON files if you want to preserve existing patient records.
- The virtual environment (`venv`) does not need to be shared. Each team member should create their own virtual environment.
- If a library installation fails, check that Python is installed correctly and that the virtual environment is activated.

## 6. Troubleshooting

**Problem: Python is not recognized**

Install Python and make sure it is added to the system PATH.

**Problem: Streamlit is not recognized**

Activate the virtual environment and run:

```powershell
python -m pip install -r requirements.txt
```

Then run:

```powershell
python -m streamlit run app.py
```

**Problem: ModuleNotFoundError**

Activate the virtual environment and reinstall the required libraries:

```powershell
python -m pip install -r requirements.txt
```

---

**WANAS Dashboard – Graduation Project**