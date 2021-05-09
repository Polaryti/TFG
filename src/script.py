import fasttext

model = fasttext.load_model(r'data\FastText\cc.ca.300.bin')

# print(model.predict("Bon dia"))

print(model.is_quantized())

print(model.get_labels())
