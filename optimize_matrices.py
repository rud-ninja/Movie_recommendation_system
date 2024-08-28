import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from data_preprocessing import data_preprocess

torch.manual_seed(9)

data = data_preprocess()

_, arr = data.transform()


device = 'cuda' if torch.cuda.is_available() else 'cpu'
chunk = 20000
size = int(np.ceil(arr.shape[0]/chunk) * np.ceil(arr.shape[1]/chunk))
n_features = 200
epochs = 20
lambda_reg = 0.01


test_P = torch.rand(arr.shape[0], n_features, requires_grad=True, device=device)
test_Q = torch.rand(n_features, arr.shape[1], requires_grad=True, device=device)
test_Q.data *= 5


optimizer = optim.AdamW(params=[test_P, test_Q], lr=0.1)



for epoch in range(epochs):
    running_loss = 0
    for i in range(0, arr.shape[0], chunk):
        P_chnk = test_P[i:i+chunk, :]
        for j in range(0, arr.shape[1], chunk):
            Q_chnk = test_Q[:, j:j+chunk]
            loss_ref = torch.tensor(arr[i:i+chunk, j:j+chunk].todense(), dtype=torch.float32)
            mask = (loss_ref > 0).int()
            loss_ref = torch.mul(loss_ref, mask).to(device)

            out = P_chnk @ Q_chnk
            loss = F.mse_loss(loss_ref, out) + lambda_reg * (torch.sum(P_chnk**2) + torch.sum(Q_chnk**2))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            running_loss += loss.item()

    avg_loss = running_loss/size
    print(f"Epoch: {epoch}; Loss: {avg_loss}")



with torch.no_grad():
    P = test_P.cpu().detach().numpy()
    Q = test_Q.cpu().detach().numpy()


np.save('D:/movie_recommendation_system/matrices/P.npy', P)
np.save('D:/movie_recommendation_system/matrices/Q.npy', Q)