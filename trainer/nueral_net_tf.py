import time
import numpy
import tensorflow as tf
from model.nueral_net_tf import MLP


def train_encoder_decoder(
    encoder: MLP,
    decoder: MLP,
    train_data: numpy.ndarray,
    train_params: dict,
) -> dict:
    """
    train_data: (ndata, nfeat)
                 it should be passed after scaling!
    """
    assert isinstance(train_data, numpy.ndarray)
    assert train_data.ndim == 2
    batch_size: int = train_params["batch_size"]
    drop_last_dl: float = train_params["drop_last_dl"]
    lr: float = train_params["lr"]
    num_epochs: int = train_params["epochs"]
    loss_fun: str = train_params["loss_fun"]
    optimizer: str = train_params["optimizer"]

    if loss_fun == "mse":
        calc_loss = tf.keras.losses.MeanSquaredError()
    else:
        raise NotImplementedError(f"loss fun {loss_fun} is not implemented")

    if optimizer == "SGD":
        opt_class = tf.keras.optimizers.SGD(learning_rate=lr)
    elif optimizer == "ADAM":
        opt_class = tf.keras.optimizers.Adam(learning_rate=lr)
    else:
        raise NotImplementedError(
            f"optimizer {optimizer} is not available"   
        )

    trainDataTF = tf.data.Dataset.from_tensor_slices(train_data)
    trainDataTF = trainDataTF.shuffle(buffer_size=train_data.shape[0])
    trainDataTF = trainDataTF.batch(batch_size, 
                                    drop_remainder=drop_last_dl)
    
    start_time = time.time()
    loss_report = {"recn_mse_train": [], 
                    "lr":[],
                    }

    def grad(x, enc, dec):
        with tf.GradientTape() as tape:
            x_ = dec(enc(x))
            assert x_.shape == x.shape
            loss_value = calc_loss(x, x_)
            dEnc, dDec = tape.gradient(loss_value, 
                    [enc.trainable_variables, dec.trainable_variables]
            )
        return loss_value, dEnc, dDec

    start_time = time.time()
    for epoch in range(num_epochs):
        avg_loss = 0.0
        num_data = 0
        for x in trainDataTF:
            num_data += x.shape[0]
            loss, dEnc, dDec = grad(x, encoder, decoder)
            opt_class.apply_gradients(zip(dEnc, encoder.trainable_variables))
            opt_class.apply_gradients(zip(dDec, decoder.trainable_variables))
            avg_loss += loss.numpy().item() * x.shape[0]
        avg_loss /= num_data
        loss_report["recn_mse_train"].append(avg_loss)
        loss_report["lr"].append(opt_class.learning_rate.numpy())
        if epoch % 10 == 0:
            print(f"Epoch {epoch}", f"Reconst Trn: {avg_loss}")
    print(f"Total Training Time: {time.time() - start_time} seconds")
    return loss_report
