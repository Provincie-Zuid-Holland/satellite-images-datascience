@echo on
pip uninstall -y dist/nso_ds_classes-1.0.0-py3-none-any.whl
del /Q dist\
python setup.py bdist_wheel
pip install dist/nso_ds_classes-1.0.0-py3-none-any.whl
