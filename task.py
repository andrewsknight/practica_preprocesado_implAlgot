from __future__ import absolute_import

import argparse
import multiprocessing as mp
import logging
import tempfile
import os

import pickle
import gensim
import pandas as pd
import numpy as np
import tensorflow as tf

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    Embedding,
    LSTM,
)
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.preprocessing import LabelEncoder


# WORD2VEC
W2V_SIZE = 300 #tamaño del embedding
W2V_WINDOW = 7 #tamaño de la ventana
# 32
W2V_EPOCH = 5 #epochs para entrrenar
W2V_MIN_COUNT = 10 #se eliminan palabras que aparecen menos de 10 veces

# KERAS
SEQUENCE_LENGTH = 300 #longitud de secuencia para el padding

# SENTIMENT definimos las etiquetas
POSITIVE = "POSITIVE" 
NEGATIVE = "NEGATIVE"
NEUTRAL = "NEUTRAL"
SENTIMENT_THRESHOLDS = (0.4, 0.7) #todas las probabilidades entre 0.4 y 0.7 se asignan a neutral

# EXPORT
KERAS_MODEL = "model.h5" #nombre del archivo del modelo
WORD2VEC_MODEL = "model.w2v" #nombre del archivo del embedding
TOKENIZER_MODEL = "tokenizer.pkl" #nombre del tokenizador
ENCODER_MODEL = "encoder.pkl" #nombre del encoder


def generate_word2vec(train_df):
  ### Genera el embedding y lo entrena
    documents = [_text.split() for _text in train_df.text.values]
    w2v_model = gensim.models.word2vec.Word2Vec(
        size=W2V_SIZE,
        window=W2V_WINDOW,
        min_count=W2V_MIN_COUNT,
        workers=mp.cpu_count(),
    )
    w2v_model.build_vocab(documents)

    words = w2v_model.wv.vocab.keys()
    vocab_size = len(words)
    logging.info(f"Vocab size: {vocab_size}")
    w2v_model.train(documents, total_examples=len(documents), epochs=W2V_EPOCH)

    return w2v_model


def generate_tokenizer(train_df):
  ### Genera nuestro tokenizador
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(train_df.text)
    vocab_size = len(tokenizer.word_index) + 1
    logging.info(f"Total words: {vocab_size}")
    return tokenizer, vocab_size


def generate_label_encoder(train_df):
  ### COdifica las etiquetas (de texto a número)
    encoder = LabelEncoder()
    encoder.fit(train_df.sentiment.tolist())
    return encoder


def generate_embedding(word2vec_model, vocab_size, tokenizer):
  ### Genera el embedding en base al Word2Vec
    embedding_matrix = np.zeros((vocab_size, W2V_SIZE))
    for word, i in tokenizer.word_index.items():
        if word in word2vec_model.wv:
            embedding_matrix[i] = word2vec_model.wv[word]
    return Embedding(
        vocab_size,
        W2V_SIZE,
        weights=[embedding_matrix],
        input_length=SEQUENCE_LENGTH,
        trainable=False,
    )


def train_and_evaluate(
    work_dir, train_df, eval_df, batch_size=1024, epochs=8, steps=1000
):

    """
    Trains and evaluates the estimator given.
    The input functions are generated by the preprocessing function.
    """

    model_dir = os.path.join(work_dir, "data/model") # Comprobamos si ya existe un modelo
    if tf.io.gfile.exists(model_dir):
        tf.io.gfile.rmtree(model_dir) #si existe lo eliminamos
    tf.io.gfile.mkdir(model_dir) #creamos un directorio de modelo

    # Configuramos donde guardar el modelo
    run_config = tf.estimator.RunConfig() 
    run_config = run_config.replace(model_dir=model_dir)

    # Nos permite seguir el entrenamiento cada 10 steps
    run_config = run_config.replace(save_summary_steps=10)

    # Generamos el Word2Vec para entrenamiento
    logging.info("---- Generating word2vec model ----")
    word2vec_model = generate_word2vec(train_df) 

    # Generamos el tokenizador con el dato de entrenamiento e inferimos en el de entrenamiento y evaluación
    logging.info("---- Generating tokenizer ----")
    tokenizer, vocab_size = generate_tokenizer(train_df)

    logging.info("---- Tokenizing train data ----")
    x_train = pad_sequences(
        tokenizer.texts_to_sequences(train_df.text), maxlen=SEQUENCE_LENGTH
    ) #Añadimos el padding
    logging.info("---- Tokenizing eval data ----")
    x_eval = pad_sequences(
        tokenizer.texts_to_sequences(eval_df.text), maxlen=SEQUENCE_LENGTH
    )

    # Generamos los encodings de las etiquetas tanto para entrenamiento como para evaluación
    logging.info("---- Generating label encoder ----")
    label_encoder = generate_label_encoder(train_df)

    logging.info("---- Encoding train target ----")
    y_train = label_encoder.transform(train_df.sentiment.tolist())
    logging.info("---- Encoding eval target ----")
    y_eval = label_encoder.transform(eval_df.sentiment.tolist())

    y_train = y_train.reshape(-1, 1) #adaptamos la forma del tensor
    y_eval = y_eval.reshape(-1, 1)

    # Create Embedding Layer
    logging.info("---- Generating embedding layer ----")
    embedding_layer = generate_embedding(word2vec_model, vocab_size, tokenizer) #capade embedding
    # Construimos nuestra red LSTM
    logging.info("---- Generating Sequential model ----")
    model = Sequential()
    model.add(embedding_layer)
    model.add(Dropout(0.5))
    model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1, activation="sigmoid"))

    model.summary()

    logging.info("---- Adding loss function to model ----")
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"]) #problema binario y medimos según el accuracy

    logging.info("---- Adding callbacks to model ----")
    callbacks = [
        ReduceLROnPlateau(monitor="val_loss", patience=5, cooldown=0), 
        EarlyStopping(monitor="val_accuracy", min_delta=1e-4, patience=5), #early stopping para agilizar
    ]

    logging.info("---- Training model ----")
    model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        steps_per_epoch=steps,
        epochs=epochs,
        validation_split=0.1,
        verbose=1,
        callbacks=callbacks,
    ) #entrenamos nuestro modelo

    logging.info("---- Evaluating model ----")
    score = model.evaluate(x_eval, y_eval, batch_size=batch_size) #evaliamos el modelo
    logging.info(f"ACCURACY: {score[1]}")
    logging.info(f"LOSS: {score[0]}")

    logging.info("---- Saving models ----")
    pickle.dump(
        tokenizer,
        tf.io.gfile.GFile(os.path.join(model_dir, TOKENIZER_MODEL), mode="wb"),
        protocol=0,
    ) #guardamos el tokenizador
    with tempfile.NamedTemporaryFile(suffix=".h5") as local_file:
        with tf.io.gfile.GFile(
            os.path.join(model_dir, KERAS_MODEL), mode="wb" #guardamos el archivo
        ) as gcs_file:
            model.save(local_file.name)
            gcs_file.write(local_file.read())

    # word2vec_model.save(os.path.join(model_dir, WORD2VEC_MODEL))

    # pickle.dump(
    #     label_encoder, open(os.path.join(model_dir, ENCODER_MODEL), "wb"), protocol=0
    # )


if __name__ == "__main__":

    """Main function called by AI Platform."""

    logging.getLogger().setLevel(logging.INFO) # Activamos el logger

    parser = argparse.ArgumentParser( #Implementamos el parseador de argumentos. Ver script de preprocesamiento para más detalle
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--job-dir",
        help="Directory for staging trainer files. "
        "This can be a Google Cloud Storage path.",
    )

    parser.add_argument(
        "--work-dir",
        required=True,
        help="Directory for staging and working files. "
        "This can be a Google Cloud Storage path.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
        help="Batch size for training and evaluation.",
    )

    parser.add_argument(
        "--epochs", type=int, default=8, help="Number of epochs to train the model",
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=1000,
        help="Number of steps per epoch to train the model",
    )

    args = parser.parse_args() #cargamos todos los argumentos antes mencionados

    train_data_files = tf.io.gfile.glob(
        os.path.join(args.work_dir, "data/transformed_data/train/part-*") # el * permite que lea todas las particiones
    ) #archivos de entrenamiento
    eval_data_files = tf.io.gfile.glob(
        os.path.join(args.work_dir, "data/transformed_data/eval/part-*")
    ) #archivos de test

    train_df = pd.concat(
        [
            pd.read_csv(
                f,
                names=["text", "sentiment"],
                dtype={"text": "string", "sentiment": "string"},
            ) # leemos cada csv 
            for f in train_data_files
        ]
    ).dropna() #generamos un dataframe con todas las particiones

    eval_df = pd.concat(
        [
            pd.read_csv(
                f,
                names=["text", "sentiment"],
                dtype={"text": "string", "sentiment": "string"},
            )
            for f in eval_data_files
        ]
    ).dropna()

    train_and_evaluate(
        args.work_dir,
        train_df=train_df,
        eval_df=eval_df,
        batch_size=args.batch_size,
        epochs=args.epochs,
        steps=args.steps,
    )
