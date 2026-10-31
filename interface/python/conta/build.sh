# auth in .pypirc
rm -rf ../conta/dist
pip install --upgrade build twine
python3 -m build 
python3 -m twine upload --verbose dist/*
