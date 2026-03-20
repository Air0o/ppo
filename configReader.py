import json

def getConfig() -> dict:
    with open("config.json") as file:
        data = json.load(file)
    return data

if __name__ == "__main__":
    print(getConfig())