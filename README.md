# 👝 **COTOMKA**: Medical CT Dataset Tools

**COTOMKA** is a simple python package for storage and pre-processing of medical computed tomography (CT) datasets. It provides unified interfaces and tools for pre-processing, saving and loading CT scans and their annotations (segmentation masks, etc.).


## Installation

1. Clone the repository and install the package:

   ```bash
   git clone https://github.com/mishgon/cotomka.git && cd cotomka && pip install -e .
   ```

2. If you encounter issues with the `opencv` library, run:

   ```bash
   pip uninstall opencv-python && pip install opencv-python-headless
   ```


## Configuration

Specify the root directory for datasets by creating a config file at `~/.config/cotomka/cotomka.yaml` with the following content:

```yaml
root_dir: /path/to/your/dataset/directory
```

Replace `/path/to/your/dataset/directory` with your desired location.


## Usage

For a usage example, check out the [usage_example.ipynb](usage_example.ipynb) notebook.


## License

**COTOMKA** is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.