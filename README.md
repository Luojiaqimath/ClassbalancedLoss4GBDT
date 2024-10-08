This is a **Python package** containing class-balanced loss functions for gradient boosting decision tree. 
Please refer to the paper "Improving GBDT Performance on Imbalanced Datasets: An Empirical Study of Class-Balanced Loss Functions" for more details.

You can install the package by:
```python
pip install gbdtCBL==0.1
```

To use XGBoost, you need to first install the following packages:
```python
pip install xgboost
```

To use LightGBM, you need to first install the following packages:
```python
pip install lightgbm --config-settings=cmake.define.USE_GPU=ON
```

To use SketchBoost, you need to first install the following packages:
```python
pip install -U cupy-cuda11x py-boost  # for cuda 11.x

pip install -U cupy-cuda12x py-boost  # for cuda 12.x
```


Some examples are provided in the Examples file. Datasets used in the paper are also provided.



**Feel free to reach out if you have any ideas or questions!**
