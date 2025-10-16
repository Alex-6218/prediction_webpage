const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
ctx.lineWidth = 12;
ctx.lineCap = "round";
ctx.strokeStyle = "black";
// Ensure the canvas bitmap has a white background (CSS alone doesn't affect pixel data)
ctx.fillStyle = "white";
ctx.fillRect(0, 0, canvas.width, canvas.height);
let drawing = false;
canvas.addEventListener("mousedown", e => {
  drawing = true;
  const rect = canvas.getBoundingClientRect();
  // map mouse coordinates from CSS pixels to canvas bitmap pixels
  const x = (e.clientX - rect.left) * (canvas.width / rect.width);
  const y = (e.clientY - rect.top) * (canvas.height / rect.height);
  ctx.beginPath();
  ctx.moveTo(x, y);
});

canvas.addEventListener("mousemove", e => {
  if (!drawing) return;
  const rect = canvas.getBoundingClientRect();
  const x = (e.clientX - rect.left) * (canvas.width / rect.width);
  const y = (e.clientY - rect.top) * (canvas.height / rect.height);
  ctx.lineTo(x, y);
  ctx.stroke();
});

canvas.addEventListener("mouseup", () => {
  drawing = false;
  try { ctx.closePath(); } catch (e) {}
});

canvas.addEventListener("mouseout", () => {
  drawing = false;
  try { ctx.closePath(); } catch (e) {}
});

document.getElementById("clear").onclick = () => {
  ctx.clearRect(0,0,canvas.width,canvas.height);
  // Refill white so the bitmap is white (not transparent)
  ctx.fillStyle = "white";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  document.getElementById("result").textContent = "";
};

document.getElementById("submit").onclick = async () => {
  const small = document.createElement("canvas");
  small.width = small.height = 28;
  const sctx = small.getContext("2d");
  // Fill background white so transparent areas become white (like MNIST)
  sctx.fillStyle = "white";
  sctx.fillRect(0, 0, small.width, small.height);
  // Draw scaled version of the main canvas
  sctx.drawImage(canvas, 0, 0, 28, 28);
  // Debug: log a small sample of pixels to verify content
  const imgData = sctx.getImageData(0, 0, 28, 28).data;
  // Convert to grayscale values (0-255) and log first 10 values
  const gray = [];
  for (let i = 0; i < imgData.length; i += 4) {
    // RGBA -> grayscale (simple average)
    gray.push(Math.round((imgData[i] + imgData[i+1] + imgData[i+2]) / 3));
    if (gray.length >= 10) break;
  }
  console.log('small canvas sample gray pixels:', gray);
  const dataURL = small.toDataURL("image/png");

  // --- Additional debug: sample the main canvas pixels too ---
  try {
    const mainImg = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
    const mainGray = [];
    for (let i = 0; i < mainImg.length; i += 4) {
      mainGray.push(Math.round((mainImg[i] + mainImg[i+1] + mainImg[i+2]) / 3));
      if (mainGray.length >= 10) break;
    }
    console.log('main canvas sample gray pixels:', mainGray);
  } catch (err) {
    console.warn('Could not read main canvas pixels:', err);
  }

  // Show the small canvas on the page (temporary debug) so you can see exactly what's sent
  small.style.border = '1px solid #444';
  small.style.margin = '8px';
  small.style.imageRendering = 'pixelated';
  small.id = 'debug-small-canvas';
  // Remove previous debug canvas if present
  const prev = document.getElementById('debug-small-canvas');
  if (prev) prev.remove();
  document.body.appendChild(small);

  const res = await fetch("/classify", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ image: dataURL })
  });
  const data = await res.json();
  console.log('Received data:', data);  // Debug log
  const probs = data.probs[0];  // Get first row of probabilities
  document.getElementById("result").textContent =
    `Prediction: ${data.prediction}\nProbs: ${probs.map(p=>Number(p).toFixed(2)).join(", ")}`;
};
