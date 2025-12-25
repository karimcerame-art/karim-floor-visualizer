// ===============================
// ELEMENT REFERENCES
// ===============================
const roomInput = document.getElementById("room-input");
const laminateInput = document.getElementById("laminate-input");
const applyButton = document.getElementById("apply-floor");
const previewTop = document.getElementById("preview-top");
const previewBottom = document.getElementById("preview-bottom");

// ===============================
// STATE
// ===============================
let roomImage = null;
let laminateImage = null;

// ===============================
// LOAD ROOM IMAGE
// ===============================
roomInput.addEventListener("change", (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const img = new Image();
    img.onload = () => {
        roomImage = img;
        previewTop.innerHTML = "";
        previewTop.appendChild(img);
        img.style.width = "100%";
        img.style.borderRadius = "16px";
    };
    img.src = URL.createObjectURL(file);
});

// ===============================
// LOAD LAMINATE IMAGE
// ===============================
laminateInput.addEventListener("change", (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const img = new Image();
    img.onload = () => {
        laminateImage = img;
    };
    img.src = URL.createObjectURL(file);
});

// ===============================
// APPLY FLOOR (NO AI – STABLE)
// ===============================
applyButton.addEventListener("click", () => {
    if (!roomImage || !laminateImage) {
        alert("Please upload room image and laminate texture");
        return;
    }

    const canvas = document.createElement("canvas");
    const ctx = canvas.getContext("2d");

    canvas.width = roomImage.naturalWidth;
    canvas.height = roomImage.naturalHeight;

    // Draw room
    ctx.drawImage(roomImage, 0, 0);

    // Define floor area (bottom 45%)
    const floorStartY = canvas.height * 0.55;

    // Tile laminate
    const pattern = ctx.createPattern(laminateImage, "repeat");
    ctx.globalAlpha = 0.95;
    ctx.fillStyle = pattern;
    ctx.fillRect(0, floorStartY, canvas.width, canvas.height - floorStartY);
    ctx.globalAlpha = 1.0;

    // Show result
    const resultImg = new Image();
    resultImg.src = canvas.toDataURL("image/png");
    resultImg.style.width = "100%";
    resultImg.style.borderRadius = "16px";

    previewBottom.innerHTML = "";
    previewBottom.appendChild(resultImg);
});
