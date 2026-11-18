# NumPy references

One module per kernel, named `reference/<name>.py`. The function is the math
and the oracle that Metal launches are checked against.

`_template.py` shows the file shape. It is not a reference implementation.
Copy it, then replace the body. NumPy only — the Metal hosts live elsewhere.

Install NumPy with the `reference` or `dev` extra of the Python package:

```bash
python3 -m pip install -e "./python[reference]"
```
