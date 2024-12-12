from flask import Blueprint, jsonify, request, send_file
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from Bio import SeqIO
import torch
import torch.nn as nn

# Blueprint for API routes
bp = Blueprint('api', __name__)

# Neural network for trait propagation
class GeneFlowNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GeneFlowNN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )
    
    def forward(self, x):
        return self.fc(x)

# Load pre-trained model or initialize
gene_flow_model = GeneFlowNN(input_dim=3, output_dim=1)
gene_flow_model.eval()

@bp.route('/')
def home():
    return jsonify({
        "message": "Welcome to GeneFlow Predictor™ API",
        "endpoints": {
            "/api/process": "POST - Upload files, run simulation, and generate heatmap"
        }
    })

# Unified endpoint for uploading files, running simulation, and generating heatmap
@bp.route('/api/process', methods=['POST'])
def process_data():
    genetic_file = request.files.get('genetic_data.fasta')
    environmental_file = request.files.get('environmental_data.csv')

    if not genetic_file or not environmental_file:
        return jsonify({"error": "Both genetic_data.fasta and environmental_data.csv files are required"}), 400

    try:
        # Process genetic data
        sequences = list(SeqIO.parse(genetic_file.stream, "fasta"))
        sequence_lengths = [len(seq.seq) for seq in sequences]
        avg_length = np.mean(sequence_lengths)

        # Process environmental data
        environmental_data = pd.read_csv(environmental_file)
        summary = environmental_data.describe().to_dict()

        # Run simulation
        mutation_rate = 0.01
        migration_rate = 0.05
        selection_pressure = 1.0
        generations = 100
        gene_frequencies = np.zeros(generations)
        gene_frequencies[0] = 0.5  # Initial frequency
        
        for t in range(1, generations):
            gene_frequencies[t] = (
                gene_frequencies[t-1] * (1 - mutation_rate) +
                migration_rate * (1 - gene_frequencies[t-1]) +
                selection_pressure * gene_frequencies[t-1] * (1 - gene_frequencies[t-1])
            )

        # Generate heatmap
        data = np.random.rand(10, 10)
        plt.figure(figsize=(8, 6))
        plt.imshow(data, cmap='hot', interpolation='nearest')
        plt.colorbar()
        plt.title("Trait Distribution Heatmap")

        # Save heatmap to buffer
        img_buffer = BytesIO()
        plt.savefig(img_buffer, format='png')
        plt.close()
        img_buffer.seek(0)

        response = {
            "message": "Processing completed successfully",
            "genetic_data": {
                "num_sequences": len(sequences),
                "avg_sequence_length": avg_length
            },
            "environmental_data_summary": summary,
            "gene_frequencies": gene_frequencies.tolist(),
        }

        return send_file(img_buffer, mimetype='image/png', as_attachment=True, attachment_filename='heatmap.png'), 200, response

    except Exception as e:
        return jsonify({"error": str(e)}), 500
