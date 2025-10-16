const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
ctx.lineWidth = 20;
ctx.lineCap = "round";
ctx.strokeStyle = "black";
let drawing = false;

canvas.addEventListener("mousedown", e => { drawing = true; draw(e); });
canvas.addEventListener("mouseup",   () => drawing = false);
canvas.addEventListener("mouseout",  () => drawing = false);
canvas.addEventListener("mousemove", draw);

function draw(e) {
  if (!drawing) return;
  const rect = canvas.getBoundingClientRect();
  ctx.lineTo(e.clientX - rect.left, e.clientY - rect.top);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(e.clientX - rect.left, e.clientY - rect.top);
}

document.getElementById("clear").onclick = () => {
  ctx.clearRect(0,0,canvas.width,canvas.height);
  document.getElementById("result").textContent = "";
};

document.getElementById("submit").onclick = async () => {
  const small = document.createElement("canvas");
  small.width = small.height = 28;
  const sctx = small.getContext("2d");
  sctx.drawImage(canvas, 0, 0, 28, 28);
  const dataURL = small.toDataURL("image/png");

  const res = await fetch("/classify", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ image: dataURL })
  });
  const data = await res.json();
  document.getElementById("result").textContent =
    `Prediction: ${data.prediction}\nProbs: ${data.probs.map(p=>p.toFixed(2)).join(", ")}`;
};
