# BUAD 313 Final Project: Spotify Playlist Optimization

*by Thomaz Bonato, Senya Wong, Finny Ho, Daniel Yang*

## Project Overview

This project explores Spotify playlist optimization using mixed binary optimization models. The goal is to generate personalized playlists that maximize user satisfaction by balancing song ratings, artist diversity, and music discovery (exploration).

The project includes:
* **Base Model**: A simple model maximizing user ratings with constraints on length and artist repetition.
* **Enriched Model**: An improved model (`enriched_model.ipynb`) that refines the base model by:
    * Relaxing artist song restrictions.
    * Increasing playlist length to 100 songs.
    * Incorporating exploratory songs using collaborative filtering (similar users, next best genre) and unrated songs.
    * Adding variety through song ordering ("peaks") and artist popularity ("scrobbles").
* **Data Wrangling & EDA**: (`eda.ipynb`) Includes tag consolidation, popularity filtering, and rating normalization.
* **Collaborative Filtering**: (`utils.py`) Custom functions to find similar users and recommend genres.
* **Analysis**: Includes critiques of both models and a robustness/sensitivity analysis of the enriched model's parameters.

## Setup

1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/danielyangdev/BUAD-313-Final-Project.git
    cd BUAD-131-Final-Project
    ```

2.  **Create a Virtual Environment (Recommended)**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install Required Packages**:
    ```bash
    pip install pandas numpy matplotlib gurobipy
    ```
    *(Note: You need a Gurobi license installed and configured for `gurobipy` to work. Academic licenses are available.)*

4.  **Download Data**:
    * Download the data files from the following Google Drive link:
        [Project Data](https://drive.google.com/drive/folders/1MYJjsgcmcy_QV8p87mFASR1UTfZsSnhI?usp=sharing)
    * Ensure you download `artists_with_genres.csv` and `songs_with_normalized_ratings.csv`.

5.  **Place Data**:
    * Place the downloaded `.csv` files inside the `data` folder.

## Running the Code

1.  **Launch Jupyter**:
    * Ensure your virtual environment is activated.
    * Run `jupyter notebook` or `jupyter lab` in your terminal from the repository's root directory.

2.  **Open Notebooks**:
    * Navigate to and open the notebooks in the Jupyter interface:
        * `eda.ipynb` (for data exploration and wrangling steps)
        * `base_model.ipynb` (for the base optimization model - *if applicable*)
        * `enriched_model.ipynb` (for the main enriched optimization model and analysis)

3.  **Run Cells**: Execute the cells in the desired notebook sequentially. The `enriched_model.ipynb` contains the primary optimization logic and analysis.