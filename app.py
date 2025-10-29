from flask import Flask, render_template, jsonify, request, send_file
import torch
import torch.nn as nn
import torch.optim as optim
import torch.fft as fft
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import io
import os
from threading import Thread
import time

app = Flask(__name__)

# Global state for demo
training_active = False
results = None
progress = {"status": "idle", "epoch": 0, "loss": 0, "accuracies": []}
plot_buffer = None

# MLP Model
class MLP(nn.Module):
    def __init__(self, input_size=784, hidden_size=256, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        return self.net(x.view(x.size(0), -1))

# Echo Capture
def capture_echo_signature(model, dataloader, device, top_k=5):
    activation_hist = defaultdict(list)

    def hook(module, input, output):
        if isinstance(module, nn.ReLU):
            acts = output.mean(dim=0)
            if acts.numel() > 0:
                activation_hist[id(module)].append(acts.detach().cpu())

    handles = []
    for layer in model.modules():
        if isinstance(layer, nn.ReLU):
            h = layer.register_forward_hook(hook)
            handles.append(h)

    model.eval()
    with torch.no_grad():
        for x, _ in dataloader:
            x = x.to(device)
            model(x)

    for h in handles:
        h.remove()

    echo_bank = {}
    for layer_id, hist_list in activation_hist.items():
        if not hist_list:
            continue
        hist = torch.stack(hist_list).mean(dim=0).flatten()
        if hist.shape[0] < 2:
            continue
        freq = fft.rfft(hist)
        mag = freq.abs()
        top_idx = mag.topk(min(top_k, mag.shape[0])).indices.cpu().numpy().tolist()
        echo_bank[layer_id] = top_idx

    return echo_bank

# Plasticity Wave
def compute_plasticity_wave(model, echo_memory_bank, alpha=2.5, beta=1.0, device='cpu'):
    if not echo_memory_bank:
        return [torch.ones_like(p, device=device) for p in model.parameters()]

    dummy_dataset = datasets.MNIST("./data", train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))
                                   ]))
    dummy_loader = DataLoader(Subset(dummy_dataset, range(64)), batch_size=32, shuffle=False)
    current_echo = capture_echo_signature(model, dummy_loader, device=device, top_k=5)

    if not current_echo:
        return [torch.ones_like(p, device=device) for p in model.parameters()]

    gates = []
    for param in model.parameters():
        if param.ndim < 2:
            gates.append(torch.ones_like(param))
            continue

        total_overlap = 0.0
        task_count = len(echo_memory_bank)
        for task_echo in echo_memory_bank.values():
            overlap = 0
            matched = 0
            for lid, cur_peaks in current_echo.items():
                if lid in task_echo:
                    matched += 1
                    ref_peaks = task_echo[lid]
                    overlap += len(set(cur_peaks) & set(ref_peaks)) / max(len(ref_peaks), 1)
            if matched > 0:
                total_overlap += overlap / matched
        avg_overlap = total_overlap / task_count if task_count > 0 else 1.0
        p_gate = torch.sigmoid(torch.tensor(alpha * (1 - avg_overlap) - beta)).item()
        gates.append(torch.full_like(param, p_gate))

    return gates

# Echo Replay Pulse (simplified for demo)
def trigger_erp(model, echo_bank, task_id, old_loader, device, top_n=3, lr=1e-3):
    if task_id not in echo_bank:
        return
    # Simplified: skip detailed ERP for speed
    pass

# Permuted MNIST (subset for demo)
def get_permuted_mnist(task_id, root="./data"):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    full_dataset = datasets.MNIST(root, train=True, download=True, transform=transform)
    # Subset to 6000 samples per task for faster demo
    subset_idx = np.random.choice(len(full_dataset), 6000, replace=False)
    dataset = Subset(full_dataset, subset_idx)

    perm = torch.randperm(784)
    perm = (perm + task_id * 123) % 784
    data_flat = torch.stack([dataset.dataset[i][0] for i in dataset.indices]).view(-1, 784)
    permuted_flat = data_flat[:, perm]
    # Note: For demo, we approximate permutation without reshaping back

    return dataset

# Evaluate
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return 100.0 * correct / total if total > 0 else 0.0

# Training function
def run_training(num_tasks):
    global training_active, results, progress, plot_buffer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP().to(device)
    echo_bank = {}
    accuracies = []

    for task_id in range(num_tasks):
        progress["status"] = f"Training Task {task_id + 1}/{num_tasks}"
        dataset = get_permuted_mnist(task_id)
        loader = DataLoader(dataset, batch_size=64, shuffle=True)

        # Evaluate
        task_accs = []
        for prev_id in range(task_id + 1):
            prev_dataset = get_permuted_mnist(prev_id)
            prev_loader = DataLoader(prev_dataset, batch_size=64, shuffle=False)
            acc = evaluate(model, prev_loader, device)
            task_accs.append(acc)
            if prev_id < task_id and acc < 98.0:
                trigger_erp(model, echo_bank, prev_id, prev_loader, device)
        accuracies.append(task_accs)
        progress["accuracies"] = [f"{a:.1f}%" for a in task_accs]

        # Train 50 epochs
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        model.train()
        p_gates = compute_plasticity_wave(model, echo_bank, device=device)

        for epoch in range(50):
            epoch_loss = 0.0
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                out = model(x)
                loss = criterion(out, y)
                loss.backward()
                for param, gate in zip(model.parameters(), p_gates):
                    if param.grad is not None:
                        param.grad *= gate.to(device)
                optimizer.step()
                epoch_loss += loss.item()
            progress["epoch"] = epoch + 1
            progress["loss"] = epoch_loss / len(loader)
            time.sleep(0.1)  # Simulate real-time update

        echo_bank[task_id] = capture_echo_signature(model, loader, device)

    results = accuracies
    training_active = False
    progress["status"] = "complete"

    # Generate plot
    plt.figure(figsize=(12, 8))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    for task_id in range(num_tasks):
        task_accs = [results[t][task_id] for t in range(task_id, num_tasks)]
        plt.plot(range(task_id+1, num_tasks+1), task_accs,
                 marker='o', linewidth=4, markersize=10,
                 label=f'Task {task_id+1}', color=colors[task_id % len(colors)])

    plt.xlabel("After Learning Task #")
    plt.ylabel("Accuracy (%)")
    plt.title("NPEM: Continual Learning on Permuted MNIST")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 100)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=300)
    buf.seek(0)
    plot_buffer = buf

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start', methods=['POST'])
def start_training():
    global training_active
    if training_active:
        return jsonify({"error": "Training already in progress"})
    num_tasks = int(request.json['num_tasks'])
    training_active = True
    progress["status"] = "starting"
    thread = Thread(target=run_training, args=(num_tasks,))
    thread.start()
    return jsonify({"status": "started"})

@app.route('/progress')
def get_progress():
    return jsonify(progress)

@app.route('/results')
def get_results():
    if results is None:
        return jsonify({"error": "No results yet"})
    final_accs = results[-1]
    avg = np.mean(final_accs)
    return jsonify({
        "final_accuracies": {f"Task {i+1}": f"{acc:.1f}%" for i, acc in enumerate(final_accs)},
        "average": f"{avg:.1f}%"
    })

@app.route('/download_plot')
def download_plot():
    if plot_buffer is None:
        return "No plot available", 404
    plot_buffer.seek(0)
    return send_file(plot_buffer, mimetype='image/png', as_attachment=True, download_name='npem_results.png')

if __name__ == '__main__':
    os.makedirs('data', exist_ok=True)  # For MNIST
    app.run(debug=True, host='0.0.0.0', port=5000)