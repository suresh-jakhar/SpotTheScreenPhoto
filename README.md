# Spot the Fake Photo
A hybrid screen-recapture detection model built to run efficiently on CPU.


---

## Development Narrative & Iteration

### First Approach: Pure Hand-crafted CV Feature Ensemble (Accuracy: 81.0%)
*   **Setup:** Captured a small dataset of 50 real photos and 50 screen photos. I built a pipeline extracting basic physical features: 2D FFT magnitude spectrum (moiré patterns), local standard deviation (sharpness), color saturation profiles, and edge distributions. These were classified via a simple Logistic Regression model.
*   **Lessons:** While extremely fast (~10ms) and explainable, 81% was insufficient. The hand-crafted thresholds failed to generalize to varying light conditions and camera distances.

### Solution: MobileNetV2 & Dataset De-biasing (Accuracy: 93.0%)
*   **Setup:** Replaced the hand-crafted classifier with a pretrained **MobileNetV2** CNN to learn deep texture representations. 
*   **The Shortcut Bias Problem:** With a small dataset, the network began learning "shortcuts." For example, if real photos contained outdoor scenes (green grass, blue sky) and screen photos were indoor recaptures, the model associated greenery and blue skies with "real", even if shown on a screen.
*   **The Dataset Trick:** To eliminate semantic shortcut learning, I captured 70 new real photos, displayed those exact images on a laptop screen, and recaptured them. The dataset now contained paired identical image contents (one real, one screen). This forced the CNN to ignore semantic objects (e.g. grass, cars) and train strictly on the physical difference in rendering. This dataset de-biasing bumped accuracy to 93.0%.

### Improvement 1: Fine-Tuning (Accuracy: 94.0%)
*   **Setup:** On new unseen test photos, the model failed on specific hardware screen types. I unfreezed the CNN backbone and ran a slow, two-stage fine-tuning process (unfreezing the features backbone at a very low learning rate of `5e-6` while training the classification head). This allowed the model to adapt its feature extraction filters, raising accuracy to 94.0%.

### Improvement 2: Hybrid CNN + LogReg Feature Ensemble (Accuracy: 95.7% → 96.8%)
*   **Setup:** The CNN struggled with high-frequency reflections and edge textures. I built a hybrid model combining the CNN predictions with hand-crafted features in a late-fusion Logistic Regression model.
*   **Iteration:** Initiating this with a small set of 8 features (FFT prominence and noise residuals) reached **95.7%**. Expanding this to the final set of **20 features**—encompassing 2D/1D Fourier periodicities, local noise auto-correlation functions (ACF) to catch subpixel pitch, and DCT row/column ratios to check for double JPEG compression artifacts—pushed the randomized test performance to **96.8%**.

---

## Honest Limitations & Vulnerabilities

Although the validation metrics are strong, the model has structural vulnerabilities due to the dataset scale:
*   **Hardware Bias:** The real photos were taken using a single smartphone camera and the screen recaptures were displayed on a single laptop screen. The model likely over-indexed on the specific subpixel layout of that screen and the sensor noise pattern of that camera.
*   **Scalability:** The current hybrid architecture is highly generalizable. If trained on a dataset containing 10+ camera sensors and 10+ screen technologies (OLED, LCD, high-density Retina displays), the L1-regularization would select more robust coefficients, ensuring reliable production deployment.

---

## Latency, Throughput & Cloud Costs

### Local Performance
*   **Latency:** **~74 ms** per image on Intel Laptop CPU (50-run average).
*   **Throughput:** **13.5 images/sec** per single CPU core.



#### Option A: Serverless (AWS Lambda: 1024 MB RAM, x86)

*   **Standby Cost:** **$0.00** 
*   **Inference Latency:** **~120 ms** (Lambda throttles CPU to ~0.6 vCPU at 1024 MB allocation).
* **Compute Duration Cost:** 120 ms × $0.0000000167/ms = $0.000002 per request.
*   **Request Fee:** \$0.20 per million requests.
*   **Total Cost:** **$2.20 per million images**.

#### Option B: Dedicated Server (AWS EC2 t3.medium: 2 vCPUs, 4 GB RAM)
*   **Fixed Standby Cost:** **~$30.00 / month** ($0.0416/hour flat rate).
*   **Maximum Throughput:** **27.0 images/sec** (utilizing both vCPUs with 2 worker processes).
*   **Practical Effective Cost:**  ~$100.00 per million images.

---

## Model Caching Trade-off

Loading the PyTorch weights (`cnn_model.pt`) from disk to memory takes **~190–300 ms**. 

*   **Without Caching (Cold Start ):**
    *   Inference time ballooning to **~374 ms** (74ms runtime + 300ms initialization).
    *   Throughput drops from 13.5 to **2.7 images/sec**.
    *   AWS Lambda billing duration increases by 5×, raising the cost to **~$11.00 per million images**.
*   **With Caching :**
    *   Model is loaded once at startup and held in memory (e.g., in a warm Lambda execution environment or a long-running FastAPI/Gunicorn process).
    *   Latency is kept at the optimal **74 ms**.
    *    For our lightweight 8.7 MB model, memory overhead is negligible, making **caching mandatory** to ensure low latency and high cost-efficiency.

---



##  Run Instructions

```bash
# Predict one image
python predict.py path/to/image.jpg

# Install dependencies
pip install -r requirements.txt
```
