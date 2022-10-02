import json

data = None

with open('dataset.json') as f:
  data = json.load(f)
  # print(len(data))


print(data)
dictionary = [
  {
    "id" : 4,
    "datanya" : "ini yang ke4"
  }
]

# # Serializing json
# json_object = json.dumps(dictionary, indent=4)

data.extend(dictionary)
json_object = json.dumps(data, indent=4)
print(json_object)

# Writing to sample.json
with open("dataset.json", "w") as outfile:
    outfile.write(json_object)