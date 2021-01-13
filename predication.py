from datetime import datetime
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error


def train(model, train_loader, optimizer, epoch, rmse_mn, mae_mn, device):
    model.train()
    avg_loss = 0.0

    for i, data in enumerate(train_loader, 0):
        # print("第 %d次训练,time:[ %s ]" % (i, datetime.now()))
        batch_u, batch_i, batch_ratings = data

        # 梯度置为0
        optimizer.zero_grad()
        loss = model.loss(batch_u.to(device), batch_i.to(device), batch_ratings.to(device))
        loss.backward(retain_graph=True)
        optimizer.step()

        avg_loss += loss.item()

        if model.beta_ema > 0.:
            model.update_ema()

        if (i + 1) % 10 == 0:
            print('%s Training: [%d epoch, %3d batch] loss: %.5f, the best RMSE/MAE: %.5f / %.5f' % (
                datetime.now(), epoch, i + 1, avg_loss / 10, rmse_mn, mae_mn))
            avg_loss = 0.0
    return 0


def test(model, test_loader, device):
    model.eval()

    if model.beta_ema > 0:
        old_params = model.get_params()
        model.load_ema_params()

    pred = []
    ground_truth = []

    for test_u, test_i, test_ratings in test_loader:
        test_u, test_i, test_ratings = test_u.to(device), test_i.to(device), test_ratings.to(device)
        scores = model(test_u, test_i)
        pred.append(list(scores.data.cpu().numpy()))
        ground_truth.append(list(test_ratings.data.cpu().numpy()))

    pred = np.array(sum(pred, []), dtype=np.float32)
    ground_truth = np.array(sum(ground_truth, []), dtype=np.float32)

    rmse = np.sqrt(mean_squared_error(pred, ground_truth))
    mae = mean_absolute_error(pred, ground_truth)

    if model.beta_ema > 0:
        model.load_params(old_params)
    return rmse, mae
