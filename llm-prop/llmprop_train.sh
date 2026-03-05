#!/usr/bin/env bash

TRAIN_PATH="data/mofclassifier2/train.csv"
VALID_PATH="data/mofclassifier2/val.csv"
TEST_PATH="data/mofclassifier2/test.csv"
EPOCHS=200
TASK_NAME="classification"
PROPERTY="CLscore"

python llmprop_train.py \
--train_data_path $TRAIN_PATH \
--valid_data_path $VALID_PATH \
--test_data_path $TEST_PATH \
--epochs $EPOCHS \
--task_name $TASK_NAME \
--property $PROPERTY \
--bs 4
