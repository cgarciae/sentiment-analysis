from pathlib import Path

import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds
import typer
from plotly import express as px
from plotly import graph_objs as go
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import cytoolz as cz
from transformers import BertTokenizer, TFBertModel
import yaml


def main(
    model_dir: Path = typer.Option(...),
    params_path: Path = typer.Option(...),
    viz: bool = False,
) -> None:

    print(f"\nmodel_dir:\n\n    {model_dir}\n")

    with open(params_path, "r") as f:
        params = yaml.safe_load(f)

    tokenizer = BertTokenizer.from_pretrained(params["bert_version"])

    BertTokenizer.prepare_for_model
    BertTokenizer.encode

    data = tfds.load("civil_comments")
    ds_train = data["train"]
    ds_train = process_dataset(ds_train, params, mode="train")

    ds_test = data["test"]
    ds_test = process_dataset(ds_test, params, mode="test")

    model = Model(params)
    model.bert.trainable = False

    for layer in model.bert.layers[-2:]:
        layer.trainable = True

    print(vars(model.bert.layers[0]))

    model(tf.zeros([1, 200], dtype=tf.int32))
    model.summary()

    return

    model.compile(
        optimizer=tf.optimizers.Adam(params["learning_rate"]),
        loss=tf.losses.binary_crossentropy,
        metrics=[tf.metrics.BinaryAccuracy()],
    )

    model.fit(
        x=ds_train,
        validation_data=ds_test,
        epochs=params["epochs"],
        steps_per_epoch=params["steps_per_epoch"],
        validation_steps=params["validation_steps"],
        callbacks=[
            tf.keras.callbacks.TensorBoard(
                log_dir=str(model_dir), update_freq=params["update_freq"]
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=f"{model_dir}/saved_model",
                monitor="val_binary_accuracy",
                mode="max",
                save_best_only=True,
            ),
        ],
    )


def process_dataset(ds, params, mode):

    tokenizer = BertTokenizer.from_pretrained(params["bert_version"])

    def encode(text):
        tokens = tokenizer.encode(
            text.decode(), max_length=params["max_length"], pad_to_max_length=True,
        )

        return np.array(tokens, dtype=np.int32)

    def parse(row):
        text = row.pop("text")

        toxicity = row["toxicity"]
        toxicity = tf.expand_dims(toxicity, axis=-1)

        [tokens] = tf.numpy_function(encode, [text], [tf.int32])
        tokens.set_shape([params["max_length"]])

        return tokens, toxicity

    ds = ds.map(parse, num_parallel_calls=params["num_parallel_calls"])
    ds = ds.batch(params["batch_size"], drop_remainder=True)

    if mode == "train":
        ds = ds.shuffle(params["buffer_size"])

    if mode != "test":
        ds = ds.repeat()

    return ds


class Model(tf.keras.Model):
    def __init__(self, params):
        super().__init__()

        self.bert = TFBertModel.from_pretrained(params["bert_version"])

        self.dense_layers = [
            tf.keras.layers.Dense(units, activation="relu")
            for units in params["layers"]
        ]

        self.output_layer = tf.keras.layers.Dense(1, activation="sigmoid")

    def call(self, inputs):

        net = self.bert(inputs)[1]

        for layer in self.dense_layers:
            net = layer(net)

        return self.output_layer(net)


if __name__ == "__main__":
    typer.run(main)
