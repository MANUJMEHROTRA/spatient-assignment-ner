from flask import Flask, render_template, request, jsonify
from transformers import BertForTokenClassification,BertTokenizerFast
import torch
import json



app = Flask(__name__)

# Load model and tokenizer
model = BertForTokenClassification.from_pretrained("./model/ner_model")
tokenizer = BertTokenizerFast.from_pretrained("./model/ner_token")


with open('./model/id_to_label.json', 'r', encoding='utf-8') as f:
    id_to_label = json.load(f)


def inference_ner(sentence):
        # Tokenize input
    inputs = tokenizer(sentence, return_tensors="pt")

    # Get predictions
    outputs = model(**inputs).logits
    predictions = torch.argmax(outputs, dim=2)
    
    print(predictions)
    # Convert predictions to labels
    predicted_labels = [id_to_label[str(label_id)] for label_id in predictions[0].tolist()]
    print(predicted_labels)
    
        # Get token words
    token_words = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    # Combine token words with their NER tags
    # ner_tags = {}
    # for token,token in list(zip(token_words, predicted_labels)):
    #     ner_tags[token_words] = token
    ner_tags = list(zip(token_words, predicted_labels))

    print(ner_tags)
    return ner_tags

        

# Define a route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define a route to handle form submission
@app.route('/process', methods=['POST'])
def process():
    sentence = request.form['sentence']
    
    # Placeholder for processing the sentence
    # Here you can add your model or processing logic
    inference  = inference_ner(sentence) 
    output = {
        "input": sentence,
        "output": f"The NER assignment is: '{inference}'"
    }
    
    return jsonify(output)


if __name__ == '__main__':
    app.run(debug=True)

