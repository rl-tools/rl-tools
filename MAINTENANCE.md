# Maintenance

- Check that the parameters exposed in the Python wrapper match the ones in the C++ loop interface
- Check if all `malloc`, `copy` and `free` copy all member structs (and recurse if nested)