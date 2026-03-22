def getParams(inputSize, outputSize, hiddenLayers, hiddenLayerSize, pyramid):
    sizes = []
    s = hiddenLayerSize
    for _ in range(hiddenLayers):
        sizes.append(s)
        if pyramid:
            s //= 2

    biases = sum(sizes) + outputSize

    all_sizes = [inputSize] + sizes + [outputSize]
    weights = sum(all_sizes[i] * all_sizes[i+1] for i in range(len(all_sizes)-1))

    return biases + weights

if __name__ == "__main__":
    print(getParams(
        inputSize=340,
        outputSize=17,
        hiddenLayers=3,
        hiddenLayerSize=1024,
        pyramid=True
    )) 