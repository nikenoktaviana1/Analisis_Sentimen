import numpy as np

def softmax(arr):
    e = np.exp(arr)
    return e / np.sum(e, axis=1, keepdims=True)

# w1=np.random.rand(4,5)
w1=np.array([[0.70506098, -0.10871205, 0.65437026, 0.72073724, 0.59730889],
            [0.07102115, 0.19979316, -0.78093917, 0.64489623, -0.26134024],
            [-0.3271392,  0.14303545, 0.87448123, -0.63874215, 0.92498628],
            [0.12994038, 0.20938897, 0.56970817, -0.60357684, 0.58286786]])
wt=np.array([1,0,0,0])
hidden_layer=np.dot(wt,w1)
print("hidden layer : " + str(hidden_layer))


# w2=np.random.rand(5,4)
w2=np.array([[0.28051035, 0.70953748, -0.32623082, 0.28924548],
            [-0.00454899, -0.92550979, -0.70937339, 0.82615607],
            [0.25331587, -0.59994193, 0.96747514, 0.31045799],
            [0.73702188, 0.71642267, 0.50797742, -0.3142267 ],
            [0.35612519, -0.28311639, -0.40213934, 0.09657033]])
hidden_layer=np.dot(wt,w1),
output_layer=np.dot(hidden_layer,w2)
print("output layer : " + str(output_layer))


nilai_softmax=softmax(output_layer)
print("nilai softmax : " + str(nilai_softmax))


#y_pred softmax  wc=+1 Token Diff (wc=+1)
wc1=np.array([0,1,0,0])
wc2=np.array([0,0,1,0])

Diff1=nilai_softmax-wc1
print("Diff 1 : " + str(Diff1))

Diff2=nilai_softmax-wc2
print("Diff 2 : " + str(Diff2))

SumOfDiff=Diff1+Diff2
print("Sum of Diff : " + str(SumOfDiff))

deltaW2=np.outer(hidden_layer,SumOfDiff)
print("Delta for W2 : " + str(deltaW2))

EI=np.array([0.7800084,-0.55105666, -0.52777118, 0.29881944])
W2EI=np.dot(w2,EI)
print("nilai W2EI : " +str(W2EI))

deltaW1=np.outer(wt,W2EI)
print("Delta for W1 : " + str(deltaW1))


#Update Weight
updateW1=w1-0.025*deltaW1
print("Update W1 : " + str(updateW1))



