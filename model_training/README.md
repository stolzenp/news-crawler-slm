
# Model Training

---

## Files

| File                     | Description                                                                                |
|--------------------------|--------------------------------------------------------------------------------------------|
| `finetune_model.py`      | Script for fine-tuning a small language model (SLM)                                        |
| `custom_hf_instances.py` | File containing custom classes and functions of HuggingFace instances used for fine-tuning |


---

## Commands

### `finetune_model.py`

```bash
python -m model_training.finetune_model [options]
```

**Arguments**:

* `-t`, `--target`  
  *Choices:* `plaintext`, `json`  
  Description: Specify the target output format.

* `-cl`, `--contrastive-loss`  
  *Flag*  
  Description: Enable training with contrastive loss.

* `-r`, `--resume`  
  *String* (default: `None`)  
  Description: Weights & Biases run name to resume training from a previous run.