
# Evaluation

---

## 📂 Files

| File                   | Description                                               |
|------------------------|-----------------------------------------------------------|
| `evaluate_model.py`    | Script for evaluating a model                             |
| `omit_degeneration.py` | Script for filtering out results affected by degeneration |
| `upload_model.py`      | Script for uploading a model to HuggingFace Hub           |


---

## ▶️ Commands

### `evaluate_model.py`

```bash
python -m evaluation.evaluate_model [options]
```

**Arguments**:

* `-t`, `--target`  
  *Choices:* `plaintext`, `json`  
  Description: Specify the target output format.

* `-p`, `--perplexity`  
  *Flag*   
  Description: Compute perplexity on the evaluation dataset.

* `-i`, `--inference`  
  *Flag*     
  Description: Run inference-based evaluation metrics.

* `-full`, `--full-eval`  
  *Flag*    
  Description: Run all evaluation metrics (perplexity + inference).

### `omit_degeneration.py`

```bash
python -m evaluation.omit_degeneration [options]
```

**Arguments**:

* `-i`, `--input`  
  *String*  
  Description: Path to the metric scores file (in `.json` format).

### `upload_model.py`

```bash
python -m evaluation.upload_model
```