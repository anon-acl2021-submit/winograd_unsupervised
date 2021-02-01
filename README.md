# Anonymous ACL-IJCNLP submission

This repo contains supplemental materials accompanying an anonymous ACL-2021 submission. This is a temporary repository that will be replaced by a non-anonymous one after the anonymity period expires. 

To run these scripts you'll need huggingface [transformers](https://github.com/huggingface/transformers) library, any version starting from 4.0.1 should work. 
Also the code uses `nltk`, `torch`, `numpy`, and `sklearn` libraries, no version specifics.

### How can I run it?
* Use the input file dataset.tsv provided.
* To compute the MAS baseline, run 
```python calculate_MAS_score_patched.py --model bert-base-multilingual-uncased --input_file dataset.tsv > results.mbert.MAS.txt
python calculate_MAS_score_patched.py --model xlm-roberta-large --input_file dataset.tsv > results.xlmr.MAS.txt```
* To compute several unsupervised baselines, run 
```python calculate_baselines.py --model xlm-roberta-large --input_file dataset.tsv > results.xlmr.baselines.txt
python calculate_baselines.py --model bert-base-multilingual-uncased --input_file dataset.tsv > results.mbert.baselines.txt```
* To pre-calculate all the attention tensors, run
```python dump_attns.py --model bert-base-multilingual-uncased --input_file dataset.tsv --output_file dump.mbert.attn.tsv
python dump_attns.py --model xlm-roberta-large --input_file dataset.tsv --output_file dump.xlmr.attn.tsv```
* Finally, to calculate scores of the proposed method, run
```python calculate_scores_dump.py select.mbert.attn.tsv dump.mbert.scores.tsv
python calculate_scores_dump.py select.xlmr.attn.tsv dump.xlmr.scores.tsv```
