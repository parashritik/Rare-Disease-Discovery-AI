from flask import Flask, request, jsonify
from flask_cors import CORS
from inference_engine import DiscoveryEngine

app = Flask(__name__)
CORS(app) 
engine = DiscoveryEngine()

@app.route('/api/dashboard/genes', methods=['GET'])
def top_genes():
    return jsonify(engine.get_top_10_genes())

@app.route('/api/search', methods=['GET'])
def search():
    gene = request.args.get('gene')
    disease = request.args.get('disease')
    if gene: return jsonify(engine.search_by_gene(gene))
    if disease: return jsonify(engine.search_by_disease(disease))
    return jsonify({"error": "No query provided"}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)