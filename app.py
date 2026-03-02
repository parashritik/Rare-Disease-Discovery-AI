from flask import Flask, request, jsonify
from flask_cors import CORS
from inference_engine import DiscoveryEngine

app = Flask(__name__)
# Enable CORS for frontend integration (React/Vue/Static HTML)
CORS(app) 

# Initialize Discovery Engine
engine = DiscoveryEngine()

@app.route('/api/dashboard/genes', methods=['GET'])
def top_genes():
    """Endpoint for the main discovery leaderboard."""
    return jsonify(engine.get_top_10_genes())

@app.route('/api/search', methods=['GET'])
def search():
    """Dual-stream search for Genes and Diseases."""
    gene = request.args.get('gene')
    disease = request.args.get('disease')
    
    try:
        if gene:
            result = engine.search_by_gene(gene)
            if "error" in result:
                return jsonify(result), 404
            return jsonify(result)
            
        if disease:
            result = engine.search_by_disease(disease)
            if "error" in result:
                return jsonify(result), 404
            return jsonify(result)
            
    except Exception as e:
        return jsonify({"error": "Processing error", "details": str(e)}), 500

    return jsonify({"error": "No query provided. Use ?gene= or ?disease="}), 400

@app.route('/api/explain', methods=['GET'])
def explain():
    """XAI-specific endpoint for retrieving feature weights."""
    gene = request.args.get('gene')
    if not gene:
        return jsonify({"error": "Gene symbol required"}), 400
    return jsonify(engine.get_explanation(gene))

if __name__ == '__main__':
    app.run(debug=True, port=5000)