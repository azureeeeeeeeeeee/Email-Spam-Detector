from flask import Flask, url_for, render_template, request
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text

app = Flask(__name__)
model = tf.keras.models.load_model("model/model.h5")
preprocessor = hub.KerasLayer("https://kaggle.com/models/tensorflow/bert/frameworks/TensorFlow2/variations/en-uncased-preprocess/versions/1")
encoder = hub.KerasLayer("https://www.kaggle.com/models/tensorflow/bert/frameworks/TensorFlow2/variations/bert-en-uncased-l-12-h-128-a-2/versions/2", trainable=True)

def predict_spam(sentence):
  preprocessed_text = preprocessor([sentence])
  output = encoder(preprocessed_text)

  spam_prob = model.predict(output['pooled_output'])

  return spam_prob

@app.route('/')
def main():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    email = request.form['email']

    output = predict_spam(email)[0][0]

    output = round(output*100, 2)

    return render_template('index.html', output=output)

if __name__ == '__main__':
    app.run(debug=True)