import yaml

def getConfig(path = None) -> dict:
    path = path if path is not None else "config.yaml"
    with open(path, "r") as file:
        data = yaml.load(file, Loader=yaml.FullLoader)
    return data

if __name__ == "__main__":
    print(getConfig())