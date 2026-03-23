import yaml

def getConfig() -> dict:
    with open("config.jaml", "r") as file:
        data = yaml.load(file, Loader=yaml.FullLoader)
    return data

if __name__ == "__main__":
    print(getConfig())