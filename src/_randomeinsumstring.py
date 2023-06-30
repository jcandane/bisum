import numpy as np

def random_einsum_string(max_sample=6, lengthA=None, lengthB=None, greek=False, return_shapes=False, rhs=None):
    """ needs: numpy
    Generate a random string for use in einsum
    GIVEN : *lengthA (dimension of tensor A)
            *lengthB (dimension of tensor B)
    GET   : str
    """
    letters="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

    if greek:
        greek="ΑαΒβΓγΔδΕεΖζΗηΘθΙιΚκΛλΜμΝνΞξΟοΠπΡρΣσςΤτΥυΦφΧχΨψΩω"
        letters+=greek
    if lengthA==None:
        lengthA=np.random.randint(1,max_sample)
    if lengthB==None:
        lengthB=np.random.randint(1,max_sample)
    if rhs==None:
        rhs=np.random.choice([False, True])

    sample=[np.random.choice(list(letters)) for i in range(max_sample)]
    sample=np.unique(sample)
    shapes=np.random.randint(2, 10, [len(sample)])
    word1=""
    shap1=[]
    for i in range(lengthA):
        arg = np.random.choice(np.arange(len(sample)))
        word1 += sample[arg]
        shap1.append(shapes[arg])
    word2=""
    shap2=[]
    for i in range(lengthB):
        arg = np.random.choice(np.arange(len(sample)))
        word2 += sample[arg]
        shap2.append(shapes[arg])
    if rhs:
        rhs = np.random.choice(list(word1+word2))
        if return_shapes:
            return word1 + "," + word2 + " -> " + rhs, shap1, shap2
        else:
            return word1 + "," + word2 + " -> " + rhs
    else:
        if return_shapes:
            return word1 + "," + word2, shap1, shap2
        else:
            return word1 + "," + word2

## TEST
#print(random_einsum_string(rhs=False))