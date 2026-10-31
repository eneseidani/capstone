# Fix PyTorch dependency and Windows `overnight.log` test failure

The five-unit-test run failed because the project environment lacks PyTorch, and an open FileHandler prevents Windows from cleaning up the temporary test directory during exceptions.

1. Extract these files to `D:\Capstone` **overwriting only** `S05_random_subset_overnight.py` and `test_random_subset_overnight.py`.
2. Install PyTorch in the **same virtual environment**:

   ```powershell
   cd D:\Capstone
   .\.venv\Scripts\python.exe -m pip install torch
   .\.venv\Scripts\python.exe -c "import torch; print(torch.__version__)"
   ```

3. Run tests:

   ```powershell
   .\.venv\Scripts\python.exe -m unittest -v test_random_subset_overnight.py
   ```

4. Run the existing experiment:

   ```powershell
   .\.venv\Scripts\python.exe -u .\S05_random_subset_overnight.py --iterations 12 --subset-min 8 --subset-max 10 --fold-years 2018 2020 2022 --mlp-epochs 35 --workers 4 --final-2023
   ```

If missing dependencies other than torch appear, install the package requirements from your original experiment (scikit-learn, lightgbm, catboost, xgboost, numpy, pandas). The script preserves resume behavior and does not change the random-subset protocol.

The full integration test and five other tests passed in our Linux Python environment with PyTorch installed; Windows installation still needs to be tested locally.
