---
dataset_info:
  features:
  - name: label
    dtype:
      class_label:
        names:
          '0': all-domains
          '1': it-domain
  - name: images
    list: image
  - name: text
    dtype: string
  splits:
  - name: train
    num_bytes: 115281378.25917527
    num_examples: 1585
  download_size: 114806537
  dataset_size: 115281378.25917527
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
---

Extracted lists of pages from PDF resumes and the PDF texts.

Created using this code:

```python
import io
import PIL.Image
from datasets import load_dataset

def render(pdf):
    images = []
    for page in pdf.pages:
        buffer = io.BytesIO()
        page.to_image(height=840).save(buffer)
        images.append(PIL.Image.open(buffer))
    return images

def extract_text(pdf):
    return "\n".join(page.extract_text() for page in pdf.pages)

ds = load_dataset("d4rk3r/resumes-raw-pdf", split="train")
ds = ds.map(lambda x: {
    "images": render(x["pdf"]),
    "text": extract_text(x["pdf"])
}, remove_columns=["pdf"])
ds = ds.filter(lambda x: len(x["text"].strip()) > 0)
ds.push_to_hub("lhoestq/resumes-raw-pdf-for-ocr")
```