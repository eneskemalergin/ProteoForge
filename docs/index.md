# ProteoForge documentation

ProteoForge discovers differential proteoforms from an imputed peptide matrix and a condition design with a control. The installable package (v0.0.1) covers configuration, long-format peptide I/O, validation, and control-relative normalization. Discovery modeling, clustering, and the `discover()` API are not implemented yet.

## Reading order

1. [Configuration](config.md): experimental design, column mapping, and YAML loading
2. [Input and output](io.md): supported file formats, canonical columns, harmonization
3. [Prepare](prepare.md): `prepare()` and `prepare_from_parquet()` end to end
4. [Normalization](normalization.md): control-relative intensity transform (Module 1)
5. [PreparedDataset](prepared-dataset.md): output contract for downstream modeling

## Quick example

```python
from proteoforge import Config, prepare_from_parquet

config = Config.from_yaml_path("config.yaml")
dataset = prepare_from_parquet("peptides.parquet", config)

dataset.peptides.height
dataset.intensity_normalized.shape
```

## Project links

- [Repository README](https://github.com/eneskemalergin/ProteoForge)
- [Changelog](https://github.com/eneskemalergin/ProteoForge/blob/main/CHANGELOG.md)
- [License](https://github.com/eneskemalergin/ProteoForge/blob/main/LICENSE) (MIT)
