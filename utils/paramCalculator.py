def getParams(inputSize, outputSize, hiddenLayers, hiddenLayerSize):
    return inputSize + outputSize + hiddenLayerSize * hiddenLayers + \
        inputSize * hiddenLayerSize + \
        outputSize * hiddenLayerSize + \
        hiddenLayerSize * hiddenLayerSize * (hiddenLayers-1)

if __name__ == "__main__":
    print(getParams(
        inputSize=23,
        outputSize=7,
        hiddenLayers=2,
        hiddenLayerSize=64
    )) 