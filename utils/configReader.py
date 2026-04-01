import yaml

def getConfig(path:str | None = "config.yaml") -> dict:
    with open(path, "r") as file:
        data = yaml.load(file, Loader=yaml.FullLoader)
    return data

if __name__ == "__main__":
    print(getConfig())