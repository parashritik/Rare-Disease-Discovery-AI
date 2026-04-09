from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from inference_engine import DiscoveryEngine
import logging
import platform
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Initialize Discovery Engine
engine = None
try:
    engine = DiscoveryEngine()
    logger.info(f"Engine Online. Running on: {platform.system()}")
except Exception as e:
    logger.error(f"Startup Failure: {e}")


@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')


@app.route('/api/dashboard/genes', methods=['GET'])
def top_genes():
    if not engine:
        return jsonify({"error": "Engine Offline"}), 500
    return jsonify(engine.get_top_10_genes())


@app.route('/api/search', methods=['GET'])
def search():
    gene = request.args.get('gene')
    disease = request.args.get('disease')
    if not engine: return jsonify({"error": "Engine Offline"}), 500

    try:
        if gene:
            return jsonify(engine.search_by_gene(gene))

        if disease:
            return jsonify(engine.search_by_disease(disease))

    except Exception as e:
        return jsonify({"error": "Processing error", "details": str(e)}), 500

    return jsonify({"error": "No query provided"}), 400


@app.route('/api/status', methods=['GET'])
def status():
    return jsonify({
        "status": "online",
        "gen_ai": "Llama-3.3-70B (Groq LPU)",
        "platform": platform.system(),
        "engine_ready": engine is not None
    })


@app.route('/api/explain', methods=['GET'])
def explain():
    gene = request.args.get('gene')
    if not gene:
        return jsonify({"error": "Gene symbol required"}), 400
    if not engine:
        return jsonify({"error": "Engine Offline"}), 500
    return jsonify(engine.get_explanation(gene))


@app.route('/api/diseases', methods=['GET'])
def diseases():
    if not engine:
        return jsonify({"error": "Engine Offline"}), 500

    suggest = (request.args.get('suggest') or '').strip().lower()
    try:
        limit = int(request.args.get('limit', 20))
    except ValueError:
        limit = 20
    limit = max(1, min(limit, 100))

    all_diseases = list(engine.disease_map.keys())
    if suggest:
        matches = [d for d in all_diseases if suggest in d.lower()]
    else:
        matches = all_diseases

    matches = sorted(matches)[:limit]
    return jsonify({
        "count": len(matches),
        "suggest": suggest,
        "diseases": matches,
    })


@app.route('/api/gallery', methods=['GET'])
def gallery():
    if not engine:
        return jsonify({"error": "Engine Offline"}), 500

    try:
        top_genes = engine.get_top_10_genes()
        disease_cards = []

        for disease_name, genes in engine.disease_map.items():
            safe_genes = genes or []
            disease_cards.append({
                "disease": disease_name,
                "gene_count": len(safe_genes),
                "sample_genes": safe_genes[:5],
            })

        disease_cards = sorted(
            disease_cards,
            key=lambda item: item["gene_count"],
            reverse=True,
        )[:20]

        return jsonify({
            "genes": top_genes,
            "diseases": disease_cards,
            "gene_count": len(top_genes),
            "disease_count": len(disease_cards),
        })
    except Exception as e:
        return jsonify({"error": "Failed to load gallery", "details": str(e)}), 500


if __name__ == '__main__':
    host = os.getenv('FLASK_HOST', '127.0.0.1')
    port = int(os.getenv('FLASK_PORT', '8000'))
    app.run(debug=True, host=host, port=port)