import numpy as np


units=5
# W=np.random.randn(5, 20)
W=np.array([[-0.08384067,  0.05389094, -0.16728953, -0.04402454, -0.24824556,
              0.40812321, -0.08924346, -0.59908337,  0.13585656, -0.21089559,
              0.92604747, -0.06616451,  0.36310227, -0.79046286,  0.81499577,
             -0.6765078,   0.16697072,  0.99581742,  0.10917386, -0.02733255],
            [-0.14289487,  0.3890146,  -0.50905692,  0.01007607, -0.6325912,
              0.75411028, -0.99579461, -0.28998067,  0.52794679, -0.27336747,
             -0.55812204, -0.22692451,  0.23074913,  0.46490959,  0.8525809,
             -0.31256801, -0.41493682, -0.09581164,  -0.65348942,  0.868222],
            [ 0.32864657, -0.20383239,  0.27974045, -0.3035583 ,  0.34484547,
              0.42183167, -0.28461212, -0.26954448, -0.18463889, -0.27634445,
             -0.35916275,  0.09782788,  0.469568  ,  0.11810622,  0.38800818,
             -0.2654166 ,  0.13074106, -0.2759772 ,  0.20468484, -0.43865386],
            [-0.13876237,  0.43230888,  0.15773965, -0.25042242, -0.18264823,
             -0.07616175, -0.13537449, -0.1974054 , -0.33538216,  0.13820167,
              0.29614687,  0.04033977,  0.20984042,  0.04260591, -0.18922086,
             -0.18691759, -0.29926628,  0.31149325, -0.35591918, -0.26953852],
            [-0.7469885,   0.79080627,  0.18194573,  0.83778325,  0.04702878,
             -0.49762996,  0.79362659, -0.76840949,  0.40920501,  0.6046296,
             -0.79026545, -0.46473017, -0.54532653, -0.92516799, -0.65253428,
              0.9478469,  -0.85529163,  0.34871932,  0.86990796,  0.01563211]])

# Ux=np.random.randn(5, 20)
U=np.array([[-0.78114249,  0.20248958, -0.16181301, -0.785981,    0.09579491,
             -0.72565495, -0.197023,   -0.63358108,  0.23999763, -0.14133422,
             -0.79651941, -0.16652455,  0.44165317, -0.88569451,  0.30678813,
             -0.2424985,  -0.28964251, -0.14982719,  0.39583957,  0.24798371],
            [-0.01982183,  0.1994739 , -0.00218896, -0.3418544 ,  0.10404024,
             -0.49446265 ,-0.09840754, -0.04553243,  0.13584866,  0.12456008,
             -0.22113278, -0.11974911,  0.01106774, -0.0484896 ,  0.01709633,
             -0.07348764, -0.06738382,  0.00877949,  0.3633701 ,  0.00691418],
            [-0.02342916, -0.19378471,  0.20666523,  0.08170953,  0.05257188,
              0.26532187, -0.21847749, -0.16607277, -0.34502667, -0.05585357,
             -0.39116687,  0.00177022,  0.20922771,  0.11332781, -0.04900297,
              0.09444071,  0.42943755,  0.02961049,  0.44220633, -0.21021531],
            [ 0.15744441,  0.11748825,  0.27381027, -0.03372682, -0.29804187,
             -0.18400017, -0.01344236,  0.18728717, -0.28626925, -0.47900931,
              0.04681184, -0.22821266,  0.13414979, -0.18060955, -0.00232818,
             -0.01019727, -0.00211894, -0.37785919,  0.00822705,  0.37503898],
            [ 0.47337926, -0.67635669, -0.14493155, -0.08870842, -0.65075383,
             -0.91419699,  0.06878985, -0.32550184, -0.67543081, -0.76229993,
              0.03934869,  0.04671448,  0.24460776, -0.00362939, -0.25552842,
             -0.15103883,  0.00998165, -0.00746319, -0.06825706, -0.55064251]])

# b=np.random.randn(1, 20)
b=np.array([9.8822475e-04, -9.9665090e-04,  9.9451700e-04,  9.9469442e-04, 9.9272723e-04,
          -6.88243363e-04,  1.0009696e+00,  1.0009884e+00,  1.0009965e+00, 9.9900699e-01,
          -9.9992880e-04,  -9.9996722e-04,  9.9993369e-04,  9.9991076e-04,-9.9989446e-04,
           4.41567854e-04, -9.9684286e-04, -5.06024929e-04, 9.9475321e-04, 9.9280605e-04])


wi = W[:, :units]
wf = W[:, units: units * 2]
wc = W[:, units * 2: units * 3]
wo = W[:, units * 3:]

Ui = U[:, :units]
Uf = U[:, units: units * 2]
Uc = U[:, units * 2: units * 3]
Uo = U[:, units * 3:]

bi = b[:units]
bf = b[units: units * 2]
bc = b[units * 2: units * 3]
bo = b[units * 3:]

print("===Weight Dense Layer====")
# wy = np.random.randn(5, 2)
wy = np.array([[-0.03224891, -0.90037138],
 [-0.28031669, -1.03261221],
 [ 0.55466795,  1.18619224],
 [ 0.53400163, -0.88365553],
 [ 0.64554507, -1.05343887]])
# by = np.random.randn(2,)
by=np.array([-0.82886214,  0.44722233])
print("wy")
print(wy)
print("by")
print(by)

print("", end="\n")
print("===Weight Forget Gate====")
print("Wf :")
print(wf)
print("Uf :")
print(Uf)
print("bias :")
print(bf)

print("", end="\n")
print("===Weight Input Gate====")
print("Wi :")
print(wi)
print("Ui :")
print(Ui)
print("bias :")
print(bi)

print("", end="\n")
print("===Weight cell state====")
print("Wc :")
print(wc)
print("Uc :")
print(Uc)
print("bias :")
print(bc)

print("", end="\n")
print("===Weight Output Gate====")
print("Wo :")
print(wo)
print("Uo :")
print(Uo)
print("bias :")
print(bo)


def softmax(arr):
    e = np.exp(arr)
    return e / np.sum(e, axis=1, keepdims=True)

def cross_entropy(out, label):
    entropy = label * np.log(out + 1e-6) # to prevent log value overflow
    return -np.sum(entropy, axis=1, keepdims=True)

def sigmoid(arr):
    return 1 / (1 + np.exp(-arr))

def deriv_sigmoid(out):
    return out * (1 - out)

def tanh(arr):
    return np.tanh(arr)

def deriv_tanh(out):
    return 1 - np.square(out)


def predict(input_val):
    caches, states = LSTM_Cell(input_val)
    c, h = states[-1]
    pred = sigmoid(np.dot(h, wy) + by)
    print("Probabilitas Dense Layer")
    print(pred)
    label = np.argmax(pred[0])
    return label

HIDDEN = 5

def LSTM_Cell(input_val):
    batch_num = input_val.shape[1]
    caches = []
    states = []
    states.append([np.zeros([batch_num, HIDDEN]), np.zeros([batch_num, HIDDEN])])
    i=1
    for x in input_val:
        print("========== Orde " + str(i) +" ==================")
       
        c_prev, h_prev = states[-1]

        hf = sigmoid(np.dot(x, wf) +np.dot(h_prev,Uf)+ bf)
        print("1. forget Gate")
        print(hf)
  
        hi = sigmoid(np.dot(x, wi) +np.dot(h_prev,Ui) + bi)
        print("2. input Gate")
        print(hi)

        hc = tanh(np.dot(x, wc)+np.dot(h_prev,Uc) + bc)
        print("3. Kandidat Konteks")
        print(hc)
        
        ho = sigmoid(np.dot(x, wo)+np.dot(h_prev,Uo) + bo)
        print("4. Output Gate")
        print(ho)
        
        c = hf * c_prev + hi * hc
        print("5. Cell State Baru")
        print(c)
     
        h = ho * tanh(c)
        print("6. Output Final")
        print(h)
        print("", end="\n")

         
        states.append([c, h])
        caches.append([x, hf, hi, ho, hc])
        i=i+1
    print("Nilai H Final")
    print(h)
    return caches, states


x=np.array([[0.702900668,  -0.13690501, 0.651611363, 0.725284718, 0.580436679],[0.8127969,  2.364095,  -0.6084724,  0.6790205, -0.65735227],
            [2.7809677,  3.9450598, -1.4909214,  0.26488346,  -0.89886343],[2.928585,  2.9040217, -1.6556750,  -0.00391697, -2.49438300 ]])


img = x
img = np.reshape(img, [4,1,5])
print("", end="\n\n")
print("Input X",end="\n")
print(img)
pred = predict(img)
print("Predic class index : " + str(pred))

