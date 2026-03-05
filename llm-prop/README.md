## [Source](https://doi.org/10.48550/arXiv.2310.14029)

## Installation
You can install LLM-Prop by following these steps:
```
git clone https://github.com/vertaix/LLM-Prop.git
cd LLM-Prop
conda create -n <environment_name> requirement.txt
conda activate <environment_name>
```

## Usage
### Training LLM-Prop from scratch
```bash
python llmprop_train.py \
--train_data_path $TRAIN_PATH \
--valid_data_path $VALID_PATH \
--test_data_path $TEST_PATH \
--epochs $EPOCHS \
--task_name $TASK_NAME \
--property $PROPERTY

```
Then run ``` bash scripts/llmprop_train.sh ```

### Evaluating the pretrained LLM-Prop
```bash
python llmprop_evaluate.py \
--train_data_path $TRAIN_PATH \
--test_data_path $TEST_PATH \
--task_name $TASK_NAME \
--property $PROPERTY \
--checkpoint $CKPT_PATH
```
Then run ``` bash scripts/llmprop_evaluate.sh ```