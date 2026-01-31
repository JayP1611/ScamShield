import tensorflow as tf

def scamshield_model(tfidf_dim: int, hand_dim: int, emb_dim: int = 64) -> tf.keras.Model:
    """
    Builds a NN that outputs:
      - prob: scam probability (sigmoid)
      - embedding: intermediate representation (emb_dim)

    Inputs:
      - tfidf (sparse): shape (tfidf_dim,)
      - handcrafted (dense): shape (hand_dim,)
    """

    # TF-IDF input: sparse vector
    inp_tfidf = tf.keras.Input(
        shape = (tfidf_dim,),
        sparse = True,
        name = "tfidf"
    )

    # Handcrafted features: dense vector
    inp_hand = tf.keras.Input(
        shape = (hand_dim,),
        name = "handcrafted"
    )

    # Combine both feature sets
    x = tf.keras.layers.Concatenate(name = "concat")([inp_tfidf, inp_hand])

    # Hidden layers
    x = tf.keras.layers.Dense(256, activation = "relu", name = "dense1")(x)
    x = tf.keras.layers.Dropout(0.25, name = "drop1")(x)

    x = tf.keras.layers.Dense(128, activation = "relu", name="dense2")(x)
    x = tf.keras.layers.Dropout(0.15, name = "drop2")(x)

    # Embedding layer (useful for clustering / similarity later)
    embedding = tf.keras.layers.Dense(emb_dim, activation = "relu", name = "embedding")(x)

    # Final probability output
    prob = tf.keras.layers.Dense(1, activation = "sigmoid", name = "prob")(embedding)

    model = tf.keras.Model(
        inputs = {"tfidf": inp_tfidf, "handcrafted": inp_hand},
        outputs = {"prob": prob, "embedding": embedding},
        name = "ScamShieldNN"
    )

    return model