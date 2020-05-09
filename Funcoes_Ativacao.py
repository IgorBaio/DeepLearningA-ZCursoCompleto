import numpy as np
#Transfer function
def StepFunction(soma):
    if soma >= 1:
        return 1
    else:
        return 0

def SigmoideFunction(soma):
    return 1/(1+np.exp(-soma))

def HyperbolicTangentFunction(soma):
    return (np.exp(soma)-np.exp(-soma))/(np.exp(soma)+np.exp(-soma))

def ReLU(soma):
    if soma >= 0:
        return soma
    else:
        return 0

def LinearFunction(soma):
    return soma

def SoftmaxFunction(x):
    ex = np.exp(x)
    return ex/ex.sum()

teste = StepFunction(30)
print("Step Function:")
print(teste)
print(StepFunction(-1 ))

print("\nSigmoide Function:")
print(SigmoideFunction(0.358))
print(SigmoideFunction(-0.358))
print(SigmoideFunction(-1))
print(SigmoideFunction(1))

print("\nHyperbolic Tangent Function:")
print(HyperbolicTangentFunction(0.358))
print(HyperbolicTangentFunction(-0.358))


print("\nReLU:")
print(ReLU(0.358))
print(ReLU(-0.358))

print("\nLinear Function:")
print(LinearFunction(0.358))
print(LinearFunction(-0.358))

valores = [5.0, 2.0, 1.3]
print("\nSoftmax:")
print(SoftmaxFunction(valores))


def Soma(e1,e2,e3,p1,p2,p3):
    return (e1*p1)+(e2*p2)+(e3*p3)
print()
soma = Soma(5,2,1,0.2,0.5,0.1)
print("Soma = ",soma)
print("Step = ",StepFunction(soma))
print("Sigmoide = ", SigmoideFunction(soma))
print("Tangente = ", HyperbolicTangentFunction(soma))
print("ReLU = ",ReLU(soma))
print("Linear = ", LinearFunction(soma))


#0,49+ 0,000 + 0,0121 + 0,1024 = 0,6 / 4 = 0,15 -> MSE raiz de 0,15 = 0,38 -> RMSE 