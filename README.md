# ICET-Tutorial

Integrated Cluster Expansion Toolkit

## create conda env

```bash
mamba env create --prefix ./env --file ./environment.yml --force
```

## update conda env

```bash
mamba env update --prefix ./env --file environment.yml  --prune
```

## Additional requirements

Visuall c++ 14 or later is required. Can be downloaded from `https://aka.ms/vs/17/release/vc_redist.x64.exe`
