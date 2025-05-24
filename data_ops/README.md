
# Dataset Operations

---

## 📂 Files

| File                               | Description                                                     |
|------------------------------------|-----------------------------------------------------------------|
| `check_sample.py`                  | Script for checking a specific sample in a dataset              |
| `step_01_convert_to_hf_dataset.py` | Script for converting dataset text file to HuggingFace Dataset  |
| `step_02_split_dataset.py`         | Script for splitting dataset with isolated publishers           |
| `step_03_clean_html.py`            | Script for cleaning HTML in dataset                             |
| `step_05_filter_dataset.py`        | Script for filtering dataset based on token length distribution |
| `step_06_shrink_dataset.py`        | Script for shrinking dataset splits for faster evaluation       |


---

## ▶️ Commands

### `check_sample.py`

```bash
python -m data_ops.check_sample [options]
```

**Arguments**:

* `--dataset_dir`  
  *String*  
  Description: Dataset directory (expects splits).

* `--split`  
  *String*  
  Description: Split to check (e.g., train, val, test).

* `--sample_id`  
  *Integer*  
  Description: Sample ID to check.

* `--columns`  
  *List of strings*  
  Description: Columns to check (all if not specified).

### `step_01_convert_to_hf_dataset.py`

```bash
python -m data_ops.step_01_convert_to_hf_dataset
```

### `step_02_split_dataset.py`

```bash
python -m data_ops.step_02_split_dataset
```

### `step_03_clean_html.py`

```bash
python -m data_ops.step_03_clean_html
```

### `step_05_filter_dataset.py`

```bash
python -m data_ops.step_05_filter_dataset
```

### `step_06_shrink_dataset.py`

```bash
python -m data_ops.step_06_shrink_dataset
```