# ICET-Tutorial

Integrated Cluster Expansion Toolkit

## This package is only available on linux and mac 

## create conda env

```bash
mamba env create --prefix ./env --file ./environment.yml --force
```

## update conda env

```bash
conda env update --prefix ./env --file environment.yml  --prune
```

## Testing

To be sure everything is installed as expected do following 

```bash
curl -O https://icet.materialsmodeling.org/tests.zip
unzip tests.zip
python3 tests/main.py
```