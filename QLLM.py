# ============================================================
# Quantum LLM Semantic Similarity
# Based on:
# "Quantum LLMs Using Quantum Computing to Analyze
#  and Process Semantic Information"
# ============================================================

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

# ============================================================
# 1. 載入 Sentence Transformer
# ============================================================

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

# ============================================================
# 2. 輸入句子
# ============================================================

sentence_a = "the quick brown fox jumps over the lazy dog"
sentence_b = "the brown dog jumps over the lazy cat"

# ============================================================
# 3. 產生 Embeddings
# ============================================================

embedding_a = model.encode(sentence_a)
embedding_b = model.encode(sentence_b)

print("Original embedding dimension:", len(embedding_a))

# ============================================================
# 4. 降維（模擬論文 MRL truncation）
# ============================================================

DIM = 16

embedding_a = embedding_a[:DIM]
embedding_b = embedding_b[:DIM]

# ============================================================
# 5. L2 Normalization
# ============================================================

embedding_a = normalize([embedding_a])[0]
embedding_b = normalize([embedding_b])[0]

# ============================================================
# 6. Classical Cosine Similarity
# ============================================================

classical_similarity = np.dot(
    embedding_a,
    embedding_b
)

print("\nClassical Cosine Similarity:")
print(classical_similarity)

# ============================================================
# 7. 建立 Quantum Circuit
# ============================================================

num_qubits = DIM

qc = QuantumCircuit(num_qubits, num_qubits)

# ============================================================
# 8. Amplitude Encoding
# ============================================================

for i in range(num_qubits):

    a = embedding_a[i]
    b = embedding_b[i]

    state = np.array([a, b], dtype=np.complex128)

    # Normalize quantum state
    norm = np.linalg.norm(state)

    if norm == 0:
        state = np.array([1, 0], dtype=np.complex128)
    else:
        state = state / norm

    qc.initialize(state, i)

# ============================================================
# 9. Hadamard Gate
# ============================================================

qc.h(range(num_qubits))

# ============================================================
# 10. Measurement
# ============================================================

qc.measure(range(num_qubits), range(num_qubits))

print("\nQuantum Circuit:")
print(qc)

# ============================================================
# 11. Run Quantum Simulation
# ============================================================

simulator = AerSimulator()

shots = 4096

job = simulator.run(qc, shots=shots)

result = job.result()

counts = result.get_counts()

# ============================================================
# 12. 計算每個 qubit 的 P(0)
# ============================================================

probabilities_0 = []

for qubit in range(num_qubits):

    count_0 = 0

    for bitstring, count in counts.items():

        # Qiskit bit order reversed
        measured_bit = bitstring[::-1][qubit]

        if measured_bit == '0':
            count_0 += count

    p0 = count_0 / shots

    probabilities_0.append(p0)

# ============================================================
# 13. Quantum Similarity Estimation
# ============================================================

quantum_similarity = 0

for i in range(num_qubits):

    ai = embedding_a[i]
    bi = embedding_b[i]

    ci2 = ai**2 + bi**2

    quantum_similarity += (
        0.5 * ci2 * (2 * probabilities_0[i] - 1)
    )

print("\nQuantum Estimated Similarity:")
print(quantum_similarity)

# ============================================================
# 14. Compare Results
# ============================================================

print("\n==============================")
print("Comparison")
print("==============================")

print(f"Classical : {classical_similarity:.6f}")
print(f"Quantum   : {quantum_similarity:.6f}")

print(
    f"Difference: "
    f"{abs(classical_similarity - quantum_similarity):.6f}"
)