import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt

DATA_DIR = r"C:/Users/NHC/Documents/1python/clouds_train/cirriform clouds/img3.jpg"
IMG_SIZE = (640,640)

epoch=100
lr=0.01
losses=[]
imag=Image.open(DATA_DIR).resize(IMG_SIZE).convert("L")
img = np.array(imag)
img = np.where(img>140,1,0)
img3 = np.zeros(IMG_SIZE)
kernel = np.zeros((5,5))
for i in range(636):
    for j in range(636):
        p=kernel+img[i:i+5,j:j+5]
        if np.array_equal(p,kernel):
            img3[i,j]=0
        else:
            img3[i,j]=1
for i in range(636):
    for j in range(636):
        p=kernel+img[i:i+5,j:j+5]
        if np.all(p==1):
            img3[i,j]=1
        else:
            img3[i,j]=0
imag=np.array(imag)  
img=img3*imag+imag*(img3-1)*-1
plt.imshow(img, cmap='gray')
plt.axis('off')
plt.show()
    
# X,y = load_images(DATA_DIR,1000)
# X = X.reshape(-1,1,64,64)
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
# np.random.seed(42)
# def ReLU(x):
#     return np.maximum(0,x)
# def sigmoid(x):
#     return 1/( 1 + np.exp(-x))
# def flatten(x):
#     return x.reshape(-1 )
# w = np.random.randn(62*62,1)
# kernel = np.random.randn(3,3)
# def conv2d(X,kernel):
#     kh, kw = kernel.shape
#     h, w = X.shape
#     result = np.zeros((h-kh+1,w-kw+1))
#     for i in range(h-kh+1):
#         for j in range(w-kw+1):
#             result[i][j]=np.sum(X[i:i+kh,j:j+kh]*kernel)
#     return result
# def daoham_conv2d(X,y):
#     grad=np.zeros(3,3)
#     kh, kw= kernel.shape
#     for i in range(y.shape[0]):
#         for j in range(y.shape[1]):
#             grad += X[i:i+kh,j:j+kw] * y[i,j]
#     return grad
# def relu_derivative(x):
#     return(x>0).astype(float)

# # hàm forward propagation
# def forward(X):
#     conv=conv2d(X,kernel)
#     act=ReLU(conv)
#     flat=flatten(act)
#     out = flat @ w
#     out = np.clip(out,-500,500)
#     prob = sigmoid(out)
#     return prob,out, flat, conv

# # huấn luyệns
# for _ in range(epoch):
#     total_loss=0
#     for i in range(len(X_train)):
#         x = X_train[i][0]
#         label = y_train[i]
#         prob, out, flat, conv=forward(x)
#         loss = -(label*np.log(prob)+(1-label)*np.log(1-prob))
#         # backprop
#         dL_dout = prob - label
#         dL_dflat= w.T*dL_dout
#         dL_dw= flat * dL_dout
#         dL_dact= dL_dflat.reshape(62,62)
#         dL_dconv= relu_derivative(conv) * dL_dact
#         dL_dkernel= daoham_conv2d(x,dL_dconv)
#         #cập nhật trọng số
#         w -= lr*dL_dw
#         kernel -= lr*dL_dkernel
#         total_loss+= loss
#     avg_loss = total_loss / len(X_train)
#     print(f"Epoch {_+1} - Loss: {avg_loss:.4f}")
#     losses.append(avg_loss.item())
# # Vẽ đồ thị loss
# def predict(X):
#     prob, _, _, _ = forward(X[0])
#     return 1 if prob > 0.5 else 0

# correct = 0
# for i in range(len(X_test)):
#     pred = predict(X_test[i])
#     if pred == y_test[i]:
#         correct += 1

# acc = correct / len(X_test)
# print(f"Test Accuracy: {acc:.2f}")
# plt.plot(losses)
# plt.title("Loss over epochs")
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.grid()
# plt.show()
# for i in range(len(X_test)):
#     pred = predict(X_test[i])
#     true_label = y_test[i]
    
#     # Hiển thị ảnh
#     img = X_test[i].reshape(64, 64)  # X_test có shape (1,64,64), bỏ kênh đi
#     plt.imshow(img, cmap='gray')
#     plt.title(f"True: {'Dog' if true_label==1 else 'Cat'} - Pred: {'Dog' if pred==1 else 'Cat'}")
#     plt.axis('off')
#     plt.show()
    
#     # Nếu bạn chỉ muốn hiển thị một số ảnh đầu tiên, có thể thêm điều kiện dừng
#     # if i == 9:
#     #     break
     
