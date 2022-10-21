import torch
from torchviz import make_dot
import data_processing as proc
import model as m
import visualize as viz

# CONSTS
TRAIN_PER = 40

if __name__ == "__main__":
    # Generate and visualize sample
    f = proc.SemiContinuousSin(const_amp=False, const_freq=False)
    truth = f.f_points
    noise, y = f.gen_noise(200)
    viz.plot_data(noise, truth)
    # Prepare data and train model
    X_train_t, y_train_t, X_test_t, y_test_t, mm, ss =  \
        proc.preprocess(noise=noise, y=y, train_percent=TRAIN_PER)
    model = m.train_model("RNN",X_train_t, y_train_t, X_test_t, y_test_t, num_epochs=1001, hidden_size=2)
    # Predict
    x_val = torch.cat((X_train_t,X_test_t))
    y_val = torch.cat((y_train_t,y_test_t))
    pred, _ = m.predict(X_ss=x_val, y_mm=y_val, model=model)
    # Display model
    batch = X_train_t
    yhat = model(batch) # Give dummy batch to forward().
    make_dot(yhat, params=dict(list(model.named_parameters()))).render("rnn_torchviz", format="png")
    # Plot res
    test_pred, test_loss = m.predict(X_test_t, y_test_t, model)
    viz.plot_pred(x_pred=noise[:,0], y_pred=pred, truth=truth, test_loss=test_loss, mm=mm, percent_train=TRAIN_PER)