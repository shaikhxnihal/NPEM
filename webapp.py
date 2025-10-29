# ------------------------------------------------------------
# app.py – NPEM Continual Learning + Interactive Test Panel
# ------------------------------------------------------------
from flask import Flask, render_template, jsonify, request, send_file, Response
import torch, torch.nn as nn, torch.optim as optim, torch.fft as fft
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from collections import defaultdict
import numpy as np, matplotlib.pyplot as plt, io, os, json, time, base64
from threading import Thread
from PIL import Image, ImageOps

app = Flask(__name__)

# ------------------- GLOBAL STATE -------------------
training_active = False
results = None
progress = {"status":"idle","epoch":0,"loss":0,"accuracies":[]}
plot_buffer = None
accuracy_history = []
echo_bank_global = {}
trained_model = None
trained_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
task_permutations = {}          # keep the exact perm for each task

# ------------------- MODEL -------------------
class MLP(nn.Module):
    def __init__(self, input_size=784, hidden_size=256, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, num_classes)
        )
    def forward(self, x): return self.net(x.view(x.size(0), -1))

# ------------------- ECHO & PLASTICITY -------------------
def capture_echo_signature(model, dataloader, device, top_k=5):
    activation_hist = defaultdict(list)
    def hook(m, i, o):
        if isinstance(m, nn.ReLU):
            acts = o.mean(dim=0)
            if acts.numel(): activation_hist[id(m)].append(acts.detach().cpu())
    handles = [m.register_forward_hook(hook) for m in model.modules() if isinstance(m,nn.ReLU)]
    model.eval()
    with torch.no_grad():
        for x,_ in dataloader: model(x.to(device))
    [h.remove() for h in handles]

    echo_bank = {}
    for lid, hist in activation_hist.items():
        if not hist: continue
        h = torch.stack(hist).mean(0).flatten()
        if h.shape[0]<2: continue
        freq = fft.rfft(h); mag = freq.abs()
        top_idx = mag.topk(min(top_k,mag.shape[0])).indices.cpu().numpy().tolist()
        echo_bank[lid] = top_idx
    return echo_bank

def compute_plasticity_wave(model, echo_memory_bank, alpha=2.5, beta=1.0, device='cpu'):
    if not echo_memory_bank:
        return [torch.ones_like(p,device=device) for p in model.parameters()]

    dummy = datasets.MNIST("./data",train=True,download=True,
                transform=transforms.Compose([transforms.ToTensor(),
                                              transforms.Normalize((0.1307,),(0.3081,))]))
    dummy_loader = DataLoader(Subset(dummy,range(64)),batch_size=32,shuffle=False)
    current_echo = capture_echo_signature(model,dummy_loader,device,top_k=5)
    if not current_echo:
        return [torch.ones_like(p,device=device) for p in model.parameters()]

    gates = []
    for param in model.parameters():
        if param.ndim<2:
            gates.append(torch.ones_like(param)); continue
        total_overlap = 0.0; task_cnt = len(echo_memory_bank)
        for task_echo in echo_memory_bank.values():
            overlap = matched = 0
            for lid, cur in current_echo.items():
                if lid in task_echo:
                    matched += 1
                    ref = task_echo[lid]
                    overlap += len(set(cur)&set(ref))/max(len(ref),1)
            if matched: total_overlap += overlap/matched
        avg_overlap = total_overlap/task_cnt if task_cnt else 1.0
        p_gate = torch.sigmoid(torch.tensor(alpha*(1-avg_overlap)-beta)).item()
        gates.append(torch.full_like(param,p_gate))
    return gates

# ------------------- PERMUTED MNIST -------------------
def get_permuted_mnist(task_id, root="./data"):
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.1307,),(0.3081,))])
    full = datasets.MNIST(root,train=True,download=True,transform=transform)
    idx = np.random.choice(len(full),6000,replace=False)
    dataset = Subset(full,idx)

    # SAME permutation every time for this task
    if task_id not in task_permutations:
        perm = torch.randperm(784)
        perm = (perm + task_id*123) % 784
        task_permutations[task_id] = perm
    else:
        perm = task_permutations[task_id]

    class PermutedSubset(Subset):
        def __getitem__(self,i):
            img,label = self.dataset.dataset[self.indices[i]]
            flat = img.view(-1,784)
            perm_flat = flat[:,perm].view(1,28,28)
            return perm_flat, label
    return PermutedSubset(dataset.dataset, dataset.indices)

# ------------------- EVALUATE -------------------
def evaluate(model, loader, device):
    model.eval(); correct = total = 0
    with torch.no_grad():
        for x,y in loader:
            x,y = x.to(device),y.to(device)
            pred = model(x).argmax(1)
            correct += (pred==y).sum().item()
            total   += y.size(0)
    return 100.0*correct/total if total else 0.0

# ------------------- TRAINING LOOP -------------------
def run_training(num_tasks):
    global training_active,results,progress,plot_buffer,accuracy_history,echo_bank_global,trained_model
    device = trained_device
    model = MLP().to(device); echo_bank_global = {}; accuracy_history.clear(); results = None; trained_model = None

    for task_id in range(num_tasks):
        progress["status"] = f"Training Task {task_id+1}/{num_tasks}"
        dataset = get_permuted_mnist(task_id)
        loader  = DataLoader(dataset,batch_size=64,shuffle=True)

        # evaluate on *all* previous tasks
        task_accs = []
        for pid in range(task_id+1):
            prev = get_permuted_mnist(pid)
            prev_loader = DataLoader(prev,batch_size=64,shuffle=False)
            task_accs.append(evaluate(model,prev_loader,device))
        accuracy_history.append(task_accs.copy())
        progress["accuracies"] = [f"{a:.1f}%" for a in task_accs]

        # train current task
        opt = optim.Adam(model.parameters(),lr=1e-3)
        crit = nn.CrossEntropyLoss()
        model.train()
        gates = compute_plasticity_wave(model,echo_bank_global,device=device)

        for epoch in range(50):
            loss_sum = 0.0
            for x,y in loader:
                x,y = x.to(device),y.to(device)
                opt.zero_grad()
                out = model(x)
                loss = crit(out,y)
                loss.backward()
                for p,g in zip(model.parameters(),gates):
                    if p.grad is not None: p.grad *= g.to(device)
                opt.step()
                loss_sum += loss.item()
            progress["epoch"] = epoch+1
            progress["loss"]   = loss_sum/len(loader)
            time.sleep(0.05)

        echo_bank_global[task_id] = capture_echo_signature(model,loader,device)

    # final evaluation
    final = [evaluate(model,DataLoader(get_permuted_mnist(i),batch_size=64,shuffle=False),device)
             for i in range(num_tasks)]
    accuracy_history.append(final.copy())
    results = final
    trained_model = model
    trained_model.eval()
    progress["status"] = "complete"
    training_active = False

    # plot
    plt.figure(figsize=(12,8))
    colors = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd']
    for tid in range(num_tasks):
        accs = [accuracy_history[t][tid] for t in range(tid,len(accuracy_history))]
        plt.plot(range(tid+1,len(accuracy_history)+1),accs,
                 marker='o',linewidth=4,markersize=10,
                 label=f'Task {tid+1}',color=colors[tid%len(colors)])
    plt.xlabel("After Learning Task #"); plt.ylabel("Accuracy (%)")
    plt.title("NPEM: Continual Learning on Permuted MNIST")
    plt.legend(); plt.grid(True,alpha=0.3); plt.ylim(0,100)
    buf = io.BytesIO(); plt.savefig(buf,format='png',dpi=300,bbox_inches='tight')
    buf.seek(0); plot_buffer = buf; plt.close()

# ------------------- PREDICTION (high-confidence) -------------------
@app.route('/predict', methods=['POST'])
def predict():
    global trained_model
    if trained_model is None:
        return jsonify({"error": "Model not trained"}), 400

    data = request.get_json()
    img_b64 = data['image'].split(',')[1]
    task_id = int(data['task_id'])

    # Decode directly to 28x28 (already resized on client)
    img_bytes = base64.b64decode(img_b64)
    img = Image.open(io.BytesIO(img_bytes)).convert('L')

    # Just normalize (no resize!)
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    tensor = tf(img).unsqueeze(0).to(trained_device)

    # Apply task permutation
    perm = task_permutations[task_id]
    flat = tensor.view(1, -1)
    permuted = flat[:, perm].view(1, 1, 28, 28)

    with torch.no_grad():
        logits = trained_model(permuted)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred = int(logits.argmax(1).item())

    return jsonify({
        "digit": pred,
        "confidence": f"{probs[pred]:.1%}",
        "probabilities": {str(i): f"{p:.1%}" for i, p in enumerate(probs)}
    })
# ------------------- ROUTES -------------------
@app.route('/')
def index(): return render_template('index.html')

@app.route('/start', methods=['POST'])
def start_training():
    global training_active
    if training_active: return jsonify({"error":"Training in progress"}),400
    num = int(request.json['num_tasks'])
    if not 1<=num<=5: return jsonify({"error":"Tasks 1-5"}),400
    training_active = True
    progress.update({"status":"starting","epoch":0,"loss":0,"accuracies":[]})
    accuracy_history.clear(); echo_bank_global.clear(); task_permutations.clear()
    Thread(target=run_training, args=(num,), daemon=True).start()
    return jsonify({"status":"started"})

@app.route('/progress')          # <-- COMMA, NOT SEMICOLON
def get_progress(): return jsonify(progress)

@app.route('/results')
def get_results():
    if results is None: return jsonify({"error":"No results"}),400
    avg = np.mean(results)
    return jsonify({"final_accuracies":{f"Task {i+1}":f"{a:.1f}%" for i,a in enumerate(results)},
                    "average":f"{avg:.1f}%","history":accuracy_history})

@app.route('/download_plot')
def download_plot():
    if plot_buffer is None: return "No plot",404
    plot_buffer.seek(0)
    return send_file(plot_buffer, mimetype='image/png',
                     as_attachment=True, download_name='npem_results.png')

@app.route('/stream')
def stream():
    def ev():
        while True:
            if accuracy_history:
                yield f"data: {json.dumps(accuracy_history)}\n\n"
            else:
                yield ""                     # keep connection alive
            time.sleep(1)                    # <-- MOVED INSIDE
    return Response(ev(), mimetype="text/event-stream")

# ------------------- HTML TEMPLATE (UTF-8 safe) -------------------
os.makedirs('templates', exist_ok=True)
HTML = '''<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>NPEM Demo</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
  body{font-family:Segoe UI,sans-serif;margin:0;background:#f5f7fa}
  .container{max-width:1200px;margin:20px auto;padding:20px}
  .card{background:#fff;border-radius:12px;padding:20px;margin-bottom:20px;box-shadow:0 4px 12px rgba(0,0,0,.1)}
  h1{color:#2c3e50;text-align:center}
  .controls{display:flex;gap:10px;align-items:center;flex-wrap:wrap}
  button{padding:12px 24px;font-size:16px;border:none;border-radius:8px;cursor:pointer}
  .btn-primary{background:#3498db;color:#fff}
  .btn-primary:disabled{background:#95a5a6;cursor:not-allowed}
  .status{font-weight:bold;color:#2c3e50}
  .chip{display:inline-block;padding:6px 12px;margin:4px;border-radius:20px;background:#e0e0e0;font-size:14px}
  .progress-container{width:100%;background:#ecf0f1;border-radius:8px;overflow:hidden;margin:10px 0}
  .progress-bar{height:20px;background:#3498db;width:0%;transition:width .3s}
  #chart{width:100%;height:500px}
  .footer{text-align:center;color:#7f8c8d;font-size:14px;margin-top:30px}
</style></head><body>
<div class="container">
<h1>NPEM: Neural Plasticity Echo Memory</h1>
<p style="text-align:center;color:#7f8c8d">Continual Learning on Permuted MNIST</p>

<div class="card"><div class="controls">
  <label><strong>Tasks:</strong></label>
  <input type="range" id="taskSlider" min="1" max="5" value="3" style="width:200px">
  <span id="taskValue">3</span>
  <button id="startBtn" class="btn-primary">Start Training</button>
</div></div>

<div class="card">
  <div class="status" id="status">Status: idle</div>
  <div id="epochInfo"></div>
  <div class="progress-container"><div class="progress-bar" id="progressBar"></div></div>
  <div id="accuracies"></div>
</div>

<div class="card"><div id="chart"></div></div>

<div class="card" id="resultsCard" style="display:none">
  <h3>Final Results</h3>
  <div id="finalResults"></div>
  <button id="downloadBtn" class="btn-primary">Download Plot</button>
</div>

<div class="card" id="testCard" style="display:none">
  <h3>Test the Trained Model</h3>
  <div style="display:flex;gap:20px;flex-wrap:wrap">
    <div>
      <canvas id="drawCanvas" width="280" height="280"
              style="border:2px solid #bbb;background:#fff;cursor:crosshair"></canvas>
      <div style="margin-top:8px"><button id="clearBtn" class="btn-primary" style="padding:6px 12px">Clear</button></div>
    </div>
    <div style="flex:1;min-width:250px">
      <label><strong>Task (permutation):</strong></label>
      <select id="taskSelect" style="width:100%;padding:6px;margin:6px 0"></select>
      <button id="predictBtn" class="btn-primary" style="width:100%;margin-top:12px">Predict Digit</button>
      <div id="predictionResult" style="margin-top:16px;font-weight:bold"></div>
      <div id="probChart" style="height:180px;margin-top:12px"></div>
      <h4 style="margin-top:20px">Prediction History</h4>
      <div id="historyList" style="max-height:150px;overflow-y:auto;font-size:14px"></div>
    </div>
  </div>
</div>

<div class="footer">NPEM Demo – Real-time + Interactive Testing</div>
</div>

<script>
  // ----- ELEMENTS -----
  const taskSlider=document.getElementById('taskSlider'),taskValue=document.getElementById('taskValue'),
        startBtn=document.getElementById('startBtn'),statusEl=document.getElementById('status'),
        epochInfo=document.getElementById('epochInfo'),progressBar=document.getElementById('progressBar'),
        accuraciesEl=document.getElementById('accuracies'),resultsCard=document.getElementById('resultsCard'),
        finalResults=document.getElementById('finalResults'),downloadBtn=document.getElementById('downloadBtn'),
        testCard=document.getElementById('testCard'),taskSelect=document.getElementById('taskSelect'),
        predictBtn=document.getElementById('predictBtn'),predictionRes=document.getElementById('predictionResult'),
        probChartDiv=document.getElementById('probChart'),historyList=document.getElementById('historyList'),
        canvas=document.getElementById('drawCanvas'),clearBtn=document.getElementById('clearBtn');

  taskSlider.oninput=()=>taskValue.textContent=taskSlider.value;

  // ----- LIVE ACCURACY CHART -----
  let chartInit=false,plotData=[];
  function initChart(){
    const n=parseInt(taskSlider.value);
    plotData=Array.from({length:n},(_,i)=>({x:[],y:[],mode:'lines+markers',name:`Task ${i+1}`,line:{width:4},marker:{size:10}}));
    Plotly.newPlot('chart',plotData,{title:'Live Accuracy Over Tasks',
      xaxis:{title:'After Learning Task #'},yaxis:{title:'Accuracy (%)',range:[0,100]},
      hovermode:'x unified',legend:{x:0,y:1}},{responsive:true});
    chartInit = true;
  }

  // ----- START TRAINING -----
  startBtn.onclick=async()=>{
    if(startBtn.disabled)return;
    startBtn.disabled=true; startBtn.textContent='Training...';
    resultsCard.style.display='none'; testCard.style.display='none';
    initChart();
    const r=await fetch('/start',{method:'POST',headers:{'Content-Type':'application/json'},
                 body:JSON.stringify({num_tasks:parseInt(taskSlider.value)})});
    if(!r.ok){alert('Failed');startBtn.disabled=false;startBtn.textContent='Start Training';}
  };

  // ----- PROGRESS POLLING -----
  setInterval(async()=>{
    const p=await fetch('/progress').then(r=>r.json());
    statusEl.textContent=`Status: ${p.status}`;
    epochInfo.textContent=p.status.includes('Training')?`Epoch: ${p.epoch}/50 | Loss: ${p.loss.toFixed(4)}`:'';
    progressBar.style.width=p.status.includes('Training')?`${(p.epoch/50)*100}%`:'0%';
    accuraciesEl.innerHTML=''; p.accuracies.forEach((a,i)=>{
      const c=document.createElement('span'); c.className='chip';
      c.textContent=`Task ${i+1}: ${a}`;
      c.style.background=`hsl(${(i*360/p.accuracies.length)},70%,80%)`;
      accuraciesEl.appendChild(c);
    });
    if(p.status==='complete'){
      startBtn.disabled=false; startBtn.textContent='Start Training';
      fetchResults(); showTestPanel(parseInt(taskSlider.value));
    }
  },800);

  // ----- LIVE CHART STREAM -----
  const es=new EventSource('/stream');
  es.onmessage=e=>{
    const h=JSON.parse(e.data);
    if(!chartInit)return;
    plotData.forEach((t,tid)=>{
      t.x=[]; t.y=[];
      for(let i=tid;i<h.length;i++) if(h[i][tid]!==undefined){t.x.push(i+1); t.y.push(h[i][tid]);}
    });
    Plotly.react('chart',plotData,Plotly.d3.select('#chart').data()[0].layout);
  };

  // ----- FINAL RESULTS -----
  async function fetchResults(){
    const r=await fetch('/results').then(j=>j.json());
    let html='<p><strong>Final Accuracies:</strong></p>';
    for(const [k,v] of Object.entries(r.final_accuracies))
      html+=`<span class="chip" style="background:hsl(${parseInt(k.split(' ')[1])*70},70%,80%)">${k}: ${v}</span>`;
    html+=`<p><strong>Average: ${r.average}</strong></p>`;
    finalResults.innerHTML=html; resultsCard.style.display='block';
  }
  downloadBtn.onclick=()=>location.href='/download_plot';

  // ----- TEST PANEL -----
  function showTestPanel(n){
    taskSelect.innerHTML=''; for(let i=0;i<n;i++){
      const o=document.createElement('option'); o.value=i; o.textContent=`Task ${i+1}`; taskSelect.appendChild(o);
    }
    testCard.style.display='block';
  }

  // ----- CANVAS -----
  const ctx=canvas.getContext('2d'); ctx.lineWidth=20; ctx.lineCap='round';
  let drawing=false;
  canvas.addEventListener('mousedown',e=>{drawing=true;draw(e);});
  canvas.addEventListener('mousemove',draw);
  canvas.addEventListener('mouseup',()=>{drawing=false;});
  canvas.addEventListener('mouseout',()=>{drawing=false;});
  function draw(e){
    if(!drawing)return;
    const r=canvas.getBoundingClientRect();
    ctx.beginPath();
    ctx.moveTo(e.clientX-r.left,e.clientY-r.top);
    ctx.lineTo(e.clientX-r.left,e.clientY-r.top);
    ctx.stroke();
  }
  clearBtn.onclick=()=>{ctx.clearRect(0,0,canvas.width,canvas.height);};

  // ----- PREDICT -----
  predictBtn.onclick = async () => {
  // 1. Get canvas
  const canvas = document.getElementById('drawCanvas');
  const ctx = canvas.getContext('2d');

  // 2. Create 28x28 canvas
  const smallCanvas = document.createElement('canvas');
  smallCanvas.width = 28;
  smallCanvas.height = 28;
  const smallCtx = smallCanvas.getContext('2d');

  // 3. Draw resized (center-crop + scale)
  smallCtx.imageSmoothingEnabled = true;
  smallCtx.drawImage(canvas, 0, 0, 280, 280, 0, 0, 28, 28);

  // 4. Convert to grayscale + binarize
  const imgData = smallCtx.getImageData(0, 0, 28, 28);
  const data = imgData.data;
  for (let i = 0; i < data.length; i += 4) {
    const gray = data[i] * 0.299 + data[i+1] * 0.587 + data[i+2] * 0.114;
    const value = gray > 128 ? 255 : 0;
    data[i] = data[i+1] = data[i+2] = value;
  }
  smallCtx.putImageData(imgData, 0, 0);

  // 5. Send as PNG
  const dataUrl = smallCanvas.toDataURL('image/png');

  const payload = { image: dataUrl, task_id: parseInt(taskSelect.value) };

  const resp = await fetch('/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)
  });

  if (!resp.ok) { alert('Prediction error'); return; }
  const out = await resp.json();

  predictionRes.innerHTML = `Predicted: <strong>${out.digit}</strong> (${out.confidence})`;

  const bars = Object.entries(out.probabilities).map(([d, p]) => ({
    x: [d], y: [parseFloat(p)], type: 'bar'
  }));
  Plotly.newPlot(probChartDiv, bars, {
    title: 'Confidence per Digit',
    xaxis: { title: 'Digit' },
    yaxis: { title: 'Probability', range: [0, 1] }
  }, { responsive: true });

  const entry = `Task ${payload.task_id + 1} to ${out.digit} (${out.confidence})`;
  const li = document.createElement('div');
  li.textContent = entry;
  li.style.padding = '4px 0';
  historyList.prepend(li);
};
</script></body></html>'''

with open('templates/index.html','w',encoding='utf-8') as f: f.write(HTML)

# ------------------- RUN -------------------
if __name__=='__main__':
    os.makedirs('data',exist_ok=True)
    print("NPEM Demo – http://localhost:5000")
    app.run(host='0.0.0.0',port=5000,debug=False,threaded=True)