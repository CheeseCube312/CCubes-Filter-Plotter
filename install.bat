@echo off
echo Setting up the virtual environment...

:: Step 1: Create a virtual environment
python -m venv venv

:: Step 2: Activate the virtual environment
call venv\Scripts\activate

:: Step 3: Upgrade pip (Optional but recommended)
echo Upgrading pip...
pip install --upgrade pip

:: Step 4: Install core dependencies
echo Installing core dependencies...
pip install -r requirements.txt

:: Step 5: Run the Streamlit app
echo Starting the app...
streamlit run app.py

:: Pause to keep the window open after the app starts
pause
