from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
import string

nltk.download('punkt')
nltk.download('stopwords')

app = Flask(__name__)
CORS(app)

def preprocess_text(text):
    stop_words = set(stopwords.words('portuguese'))
    words = word_tokenize(text.lower(), language='portuguese')
    filtered_words = [word for word in words if word.isalnum() and word not in stop_words and word not in string.punctuation]
    return filtered_words

def summarize_text(text, ratio=0.2):
    sentences = sent_tokenize(text)
    word_freq = {}
    for word in preprocess_text(text):
        if word not in word_freq:
            word_freq[word] = 1
        else:
            word_freq[word] += 1

    sorted_freq = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    top_sentences = []
    num_sentences = int(len(sentences) * ratio)
    if num_sentences < 1:
        num_sentences = 1

    for i in range(num_sentences):
        top_sentences.append(sentences[i])

    return ' '.join(top_sentences)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize_endpoint():
    data = request.get_json()
    text = data.get('text', '')
    ratio = float(data.get('ratio', 0.2))  # Obter a taxa de resumo do JSON

    if not text:
        return jsonify({'error': 'No text provided'}), 400
    try:
        summary = summarize_text(text, ratio=ratio)
        return jsonify({'summary': summary})
    except ValueError as e:
        return jsonify({'error': str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
