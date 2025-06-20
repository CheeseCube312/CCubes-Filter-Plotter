I created this plotting tool for IR photography by yelling at AI for a few weeks.

Features:
- plot transmission data for camera filters
- show combined transmission curve
- calculate total opacity
- load illuminant and sensor quantum efficincy curve (generic CMOS by default)
- show sensor response curve at given illuminant
- show illuminant and QE graphs independently
- export everything as .png file for easy sharing


_______________________________________________________________

Installation:

1) Install Python 3.8 or higher. Make sure to Install to PATH (should be selectible in install wizard) https://www.python.org/

2) Run Install.bat

Install.bat will download install all the necessary python libraries (found in Requirements.txt)

After first install you can start with start.bat. It just starts the program with a virtual environment (venv)

______________________________________________________________

Adding/Removing data:

Turn graphs into .csv files unsing WebPlotDigitizer: https://apps.automeris.io/wpd4/

Use the Filter Importer for to turn Filters into the right format: https://github.com/CheeseCube312/CCube_WPB.csv_Filter-Importer
Use the QE Importer for to turn Quantum Efficiency data into the right format: https://github.com/CheeseCube312/CCube_WPB.csv_QE-Importer/tree/main
Just follow the instructions for those. :)

______________________________________________________________
Filter data format:
The program is designed to use a folder full of individual .tsv files for each filter. Row 1-4 contain filte information, row 5+ contains the transmission data. The data range is 300-1100nm, in 1nm steps. The software can handle incomplete wavelength data.

I've included a large collection of filters. Some was added manually by running transmission graphs through WebPlotDigitizer. Legal info in LICENSE.md

Have fun :) 
