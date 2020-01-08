from pathlib import Path

import cytoolz as cz
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import typer
import yaml
from plotly import express as px
from plotly import graph_objs as go
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from transformers import BertTokenizer, TFBertModel

import tensorflow_datasets as tfds


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

    if params["train_bert"]:
        model = ModelFinetunning(params)
    else:
        model = Model(params)

    model.bert.trainable = params["train_bert"]

    model(tf.zeros([3, 200], dtype=tf.int32))
    model.summary()

    model.compile(
        optimizer=tf.optimizers.Adam(params["learning_rate"]),
        loss=tf.losses.binary_crossentropy,
        metrics=[
            tf.metrics.BinaryAccuracy(),
            tf.metrics.Precision(),
            tf.metrics.Recall(),
            tf.metrics.TrueNegatives(),
            tf.metrics.TruePositives(),
        ],
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
        toxicity = tf.cast(toxicity >= params["toxicity_threshold"], tf.float32)

        [tokens] = tf.numpy_function(encode, [text], [tf.int32])
        tokens.set_shape([params["max_length"]])

        sample_weight = (
            params["sample_weight"][0] * (1 - toxicity)
            + params["sample_weight"][1] * toxicity
        )
        sample_weight = sample_weight[0]

        return tokens, toxicity, sample_weight

    if mode == "train":
        ds = ds.shuffle(params["buffer_size"])

    ds = ds.map(parse, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.batch(params["batch_size"], drop_remainder=True)

    if mode != "test":
        ds = ds.repeat()

    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)

    return ds


class ModelFinetunning(tf.keras.Model):
    def __init__(self, params):
        super().__init__()

        self.bert = TFBertModel.from_pretrained(params["bert_version"])
        self.output_layer = tf.keras.layers.Dense(1, activation="sigmoid")

    def call(self, inputs, training=True):

        net = self.bert(inputs, training=training)[1]

        print(net.shape)

        return self.output_layer(net)

    def apply_module(self, module, net):
        net = module["dense"](net)
        net = module["normalization"](net)
        net = module["activation"](net)

        return net

class Model(tf.keras.Model):
    def __init__(self, params):
        super().__init__()

        self.train_bert = params["train_bert"]
        self.bert = TFBertModel.from_pretrained(params["bert_version"])

        if not self.train_bert:
            self.dense_modules = [
                dict(
                    dense=tf.keras.layers.Dense(units),
                    normalization=tf.keras.layers.LayerNormalization(),
                    activation=tf.nn.relu,
                )
                for units in params["layers"]
            ]

            self.selection_module = dict(
                dense=tf.keras.layers.Dense(params["linear_units"]),
                normalization=tf.keras.layers.LayerNormalization(),
                activation=tf.nn.relu,
            )
            self.select = Select(5, use_scale=True)
            self.flatten = tf.keras.layers.Flatten()
            self.final_module = dict(
                dense=tf.keras.layers.Dense(params["final_units"]),
                normalization=tf.keras.layers.LayerNormalization(),
                activation=tf.nn.relu,
            )

        self.output_layer = tf.keras.layers.Dense(1, activation="sigmoid")

    def call(self, inputs, training=True):

        if self.train_bert:
            net = self.bert(inputs)[1]
        else:
            net = self.bert(inputs)[0]

        if not self.train_bert:
            for module in self.dense_modules:
                self.apply_module(module, net)

            print(net.shape)
            net = self.selection_module["dense"](net)
            print(net.shape)
            net = self.select(net)
            net = self.selection_module["normalization"](net)
            net = self.selection_module["activation"](net)
            print(net.shape)
            net = self.flatten(net)
            print(net.shape)
            net = self.apply_module(self.final_module, net)
            print(net.shape)

            # net = tf.reduce_max(net, axis=-2)

        return self.output_layer(net)

    def apply_module(self, module, net):
        net = module["dense"](net)
        net = module["normalization"](net)
        net = module["activation"](net)

        return net


class Select(tf.keras.layers.Layer):
    def __init__(self, n_elements, use_scale=False, **kwargs):
        super().__init__(**kwargs)

        self.n_elements = n_elements
        self.use_scale = use_scale

        if self.use_scale:
            self.scale = self.add_weight(
                name="scale",
                shape=(),
                initializer=tf.ones_initializer(),
                dtype=self.dtype,
                trainable=True,
            )
        else:
            self.scale = None

    def build(self, input_shape):
        self.keys = self.add_weight(
            name="keys", shape=[input_shape[-1], self.n_elements]
        )

    def call(self, inputs):

        # inputs: N x K
        # keys: K x E

        scores = tf.linalg.matmul(inputs, self.keys)  # (N x K) * (K x E) = N x E

        if self.use_scale:
            scores *= self.scale

        perm = list(range(len(inputs.shape)))
        perm[-2], perm[-1] = perm[-1], perm[-2]  # transpose last 2 dims

        scores = tf.transpose(scores, perm)  # E x N
        scores = tf.nn.softmax(scores, axis=-1)

        elements = tf.linalg.matmul(scores, inputs)  # (E x N) * (N x K) = E x K

        return elements


if __name__ == "__main__":
    typer.run(main)
