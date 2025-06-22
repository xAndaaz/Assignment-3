# Assignment-3: Expert Data Visualization Project

This assignment demonstrates advanced data visualization skills using Python's `matplotlib`, `seaborn`, and statistical analysis libraries. The project includes comprehensive analysis of two classic datasets: Iris (classification patterns) and Titanic (survival analysis), showcasing different visualization techniques and statistical methods.

## Project Structure
```
Assignment-3/
├── README.md
├── Iris dataset vizualization/
│   ├── iris_visualization.py
│   ├── correlation_heatmap.png
│   ├── pairplot.png
│   ├── violin_plots.png
│   └── pca_scatter.png
└── Titanic Survival Analysis/
    └── viz2.py
    └── terminal output.png
    └── Figure_1.png
```

## Overview

### Iris Dataset Analysis (`iris/`)
Explores flower measurement patterns across three species using:
- **Correlation heatmap**: Feature relationships
- **Pairplot**: Species separation patterns with density curves
- **Violin plots**: Distribution analysis by species
- **PCA scatter plot**: Dimensionality reduction visualization

### Titanic Dataset Analysis (`titanic/`)
Comprehensive survival analysis dashboard featuring:
- **Survival rates**: By passenger class with statistical annotations
- **Age distribution**: With KDE overlays and survival comparison
- **Correlation analysis**: Feature relationship heatmap
- **Fare analysis**: Violin plots by passenger class
- **Multi-dimensional analysis**: Gender, embarkation, family size impacts
- **Statistical modeling**: Age-class survival probability with confidence intervals

## Requirements
* Python 3.x
* Libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `scipy`
* Install all dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy
```

## Usage

### Run Iris Visualization:
```bash
cd iris/
python iris_visualization.py
```
Generates four PNG files with different visualization types.

### Run Titanic Visualization:
```bash
cd Titanic Survival Analysis/
python viz2.py
```
Displays comprehensive dashboard and prints statistical analysis.

## Key Features

### Technical Skills Demonstrated:
- **Advanced Statistical Analysis**: KDE, PCA, correlation analysis, chi-square tests
- **Professional Visualization**: Custom styling, color palettes, subplot layouts
- **Data Engineering**: Feature creation, preprocessing, outlier handling
- **Multiple Chart Types**: Heatmaps, violin plots, scatter plots, bar charts, histograms
- **Statistical Overlays**: Confidence intervals, trend lines, density estimations

### Libraries Used:
- `matplotlib`: Core plotting and advanced customization
- `seaborn`: Statistical visualizations and styling
- `pandas`: Data manipulation and analysis
- `numpy`: Mathematical operations
- `scikit-learn`: PCA and machine learning preprocessing
- `scipy`: Statistical analysis and hypothesis testing

## Notes
- Both datasets are automatically loaded (Iris from seaborn, Titanic from URL)
- Code is organized into clean, readable functions
- Professional styling with consistent color schemes
- Comprehensive statistical analysis with expert-level insights
- Ready for presentation and portfolio inclusion

## Output
- **Iris**: Four high-quality PNG visualizations
- **Titanic**: Interactive dashboard with statistical summary
- Both projects demonstrate expert-level data visualization and analysis skills
