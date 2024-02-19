import json
with open("/Users/pooneh/Documents/project/result2_allin_withInput.json", "r") as read_file:
    data = json.load(read_file)
data["analysis"].pop("harmonic", None)
data["analysis"].pop("perceptual", None)
data["analysis"].pop("signal", None)
data["analysis"].pop("yin", None)
with open("/Users/pooneh/Documents/project/result2_onlyspectral_withInput.json", "w") as write_file:
    json.dump(data, write_file, indent=2)