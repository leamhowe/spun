if you get 
```bash
ValueError: numpy.dtype size changed, may indicate binary incompatibility. Expected 96 from C header, got 88 from PyObject
```

You need to run

```bash
export PYTHONNOUSERSITE=True
```