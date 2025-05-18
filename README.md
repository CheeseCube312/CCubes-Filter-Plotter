I created this plotting tool for IR photography. No idea how to code. This was made by yelling at AI for a week.


What it does:
It lets you load transmission data for camera filters and plots them. You can toggle a combined graph and display everything in logarithmic view. There also an opacity indicator above the graph that tells you how much the filter darkens the image. 

"Apply sensor QE adjustment needs to be toggled on to get a fairly close match to real-world opacity values, stated in t-stops. It enables extra math that adjusts the stop value for sensor sensitivity at the given wavelengths. 


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
The program is designed to use a folder full of individual .tsv files for each filter. Row 1 contains header, row 2 contains the data. The data range is 300-1100nm, in 5nm steps. The software can handle incomplete wavelength data.

I've included a large collection of filters. Most was added by scraping the LeeFilter website since that data matches my swatchbook. Since I know that Lee Filters rise to ~90% transmission at 800nm and then stay there the data for them gets very roughly extrapolated by the software. 

Some was added manually by running transmission graphs through WebPlotDigitizer. 


Legal:
Wether or not transmission data is subject to copyright is unclear to me. From what I can gather it's a statement of fact about a product and thus can not be copyrighted. If a manufacturer wants to argue I'll just remove their data from the included dataset since I can't be arsed to fight this. 
This program is for non-commercial use and made available for free to give IR photographers a decent tool for DIY filter experiments.


Have fun :) 
