# PathSnap: Two-Laptop Demonstration & Testing Guide
**Team Number:** PT-045  
**Course:** UE24CS352A - Machine Learning | PES University  

---

## 1. Quick Summary: Can I test everything on ONE laptop?

**YES, 100%!**  
You can test the entire two-laptop distributed system on a single machine in seconds:
1. Open a terminal and run the API server:
   ```bash
   python -m uvicorn backend.app:app --host 0.0.0.0 --port 8000
   ```
2. Open your web browser and navigate to:
   ```
   http://localhost:8000/ui
   ```
   *(Or double-click `frontend/index.html` in your file explorer)*
3. Select any of the 10 evaluation trips, click **"Run Pipeline & Evaluate"**, and watch the live API request logs print in your terminal while the UI updates with accuracy metrics and the Folium map.

---

## 2. Is it better if hosted on Vercel?

### Honest Technical Recommendation:
**Keep the local two-laptop setup as mandated by the evaluation guidelines.**

Here is why:
1. **Evaluation Requirement:** The hackathon guidelines explicitly state:
   > *"1. The API must be running on Laptop 1.*  
   > *2. The UI must be running on Laptop 2.*  
   > *3. Hardcoded outputs or bypassing the API are not permitted, students are required to show the evaluator the api logs on the laptop receiving the inference requests on the endpoint."*
   If the API is hosted on Vercel or cloud, you cannot show the terminal logs on the physical laptop in the exam hall.
2. **Vercel Limitations:** Vercel serverless functions have a 50MB code/bundle size ceiling and cold-start timeouts. Our system loads 88,000+ nodes, 165,000+ arcs, and a trained scikit-learn Random Forest model, which exceeds standard serverless memory limits.
3. **Best Hybrid Approach (If you want Vercel for the UI):**
   You can deploy the static `frontend/` folder to Vercel (or GitHub Pages) if you want the evaluator to access the UI from any laptop or tablet without copying files. In the UI's top bar, you simply type the local IP of Laptop 1 (`http://192.168.x.x:8000`) and click "Connect". However, running it via the local Wi-Fi / hotspot is the simplest, most reliable, and 100% compliant method.

---

## 3. Step-by-Step Two-Laptop Demonstration Plan

### Step A: Connect both laptops to the same network
- Either connect both Laptop 1 and Laptop 2 to the same Wi-Fi router / campus Wi-Fi, OR
- **Recommended for fail-safe demo:** Turn on **Mobile Hotspot** on a smartphone and connect both Laptop 1 and Laptop 2 to it. This isolates you from any university firewall restrictions.

---

### Step B: Laptop 1 Setup (Model & API Host)
1. Find Laptop 1's local IP address:
   - **Windows:** Open Command Prompt and run `ipconfig`. Look for `IPv4 Address` under your Wi-Fi adapter (e.g., `192.168.43.105`).
   - **Mac/Linux:** Run `ifconfig` or `ip a` (e.g., `192.168.43.105`).
2. Start the FastAPI backend on `0.0.0.0` (all network interfaces):
   ```bash
   python -m uvicorn backend.app:app --host 0.0.0.0 --port 8000
   ```
3. Keep this terminal visible! This is where the evaluator will watch incoming HTTP requests and inference logs.

---

### Step C: Laptop 2 Setup (Evaluator UI Client)
1. Open any web browser on Laptop 2 (Chrome, Firefox, Edge, Safari).
2. Enter the URL:
   ```
   http://<LAPTOP_1_IP>:8000/ui
   ```
   *(Example: `http://192.168.43.105:8000/ui`)*
3. The dashboard will load with a green status indicator **"Laptop 1 API: Online"**.
4. Hand Laptop 2 to the evaluator:
   - The evaluator selects any of the 10 evaluation trips (`00000000`, `00000007`, `00000009`, `00000040`, `00000052`, `00000056`, `00000065`, `00000076`, `00000077`, `00000091`).
   - The evaluator clicks **"Run Pipeline & Evaluate"**.
   - Laptop 2 sends a `POST /api/match` request.
   - Laptop 1 displays:
     ```
     INFO:     192.168.43.112:51240 - "POST /api/match HTTP/1.1" 200 OK
     [API INFERENCE] Evaluated Trip 00000052: HMM Top-1 Acc=93.13%, Continuity=97.32%, Time=205.7ms
     ```
   - Laptop 2 immediately updates showing:
     - **Top-1 Accuracy:** `93.13%`
     - **Route Continuity:** `97.32%`
     - **Latency:** `~200 ms`
     - **Folium Interactive Map:** Showing GPS points, road network, reference route, and predicted route!
