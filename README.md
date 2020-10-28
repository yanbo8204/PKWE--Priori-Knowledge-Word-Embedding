# PKWE: Prior-Knowledge Word Embedding

##Running Model
Generate word embedding: 
```bash
python train.py --data_path "$DATA_PATH" --data_name coco_precomp --embed_gererate
```
Using generated word embedding to train model:
```bash
python train.py --data_path "$DATA_PATH" --data_name coco_precomp --max_violation
```
Evaluation
```python
from vocab import Vocabulary
import evaluation
evaluation.evalrank("$RUN_PATH/coco_vse++/model_best.pth.tar", data_path="$DATA_PATH", split="test")'
```
To do cross-validation on MSCOCO, pass `fold5=True` with a model trained using 
`--data_name coco`.

##Acknowledge

The visual-semantic embedding part is based on the codebase of [VSE++](https://github.com/fartashf/vsepp), the code implementation of **[VSE++: Improving Visual-Semantic Embeddings with Hard Negatives](https://arxiv.org/abs/1707.05612)***, F. Faghri, D. J. Fleet, J. R. Kiros, S. Fidler, Proceedings of the British Machine Vision Conference (BMVC),  2018. (BMVC Spotlight)*
