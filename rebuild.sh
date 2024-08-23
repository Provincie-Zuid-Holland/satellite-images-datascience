pip uninstall -y dist/nso_ds_classes-1.0.0-py3-none-any.whl
rm -rf dist/
python -m build
pip install dist/nso_ds_classes-1.0.0-py3-none-any.whl