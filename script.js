let video = document.getElementById("video");
let canvas = document.getElementById("canvas");
let ctx = canvas.getContext("2d");
let model;
let detectionFrequency = 100; // Detection every 100ms (10 FPS)

// Color map for different object classes
const colorMap = {
    person: "red",
    bicycle: "blue",
    tv:"green",
    remote: "purple",
    car: "green",
    motorcycle: "purple",
    airplane: "orange",
    bus: "yellow",
    train: "pink",
    truck: "cyan",
    boat: "magenta",
    // Add more classes and colors as needed
};

// Start webcam with reduced resolution
async function startWebcam() {
    try {
        let stream = await navigator.mediaDevices.getUserMedia({
            video: { width: 340, height: 220 }
        });
        video.srcObject = stream;
    } catch (error) {
        console.error("Error accessing webcam:", error);
    }
}

// Load the lightweight object detection model
async function loadModel() {
    // Load a lightweight model
    model = await cocoSsd.load({ base: "lite_mobilenet_v2" });
    console.log("Model loaded!");
    detectObjects(); // Start detection loop
}

// Real-time object detection loop with optimized intervals
async function detectObjects() {
    if (!model) return;

    ctx.clearRect(0, 0, canvas.width, canvas.height); // Clear the canvas
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height); // Draw video frame

    let predictions = await model.detect(video); // Detect objects

    // Draw predictions
    predictions.forEach(prediction => {
        const color = colorMap[prediction.class] || "red"; // Default to red if class not in colorMap
        ctx.beginPath();
        ctx.rect(...prediction.bbox);
        ctx.lineWidth = 2;
        ctx.strokeStyle = color;
        ctx.fillStyle = color;
        ctx.stroke();
        ctx.fillText(
            `${prediction.class} - ${Math.round(prediction.score * 100)}%`,
            prediction.bbox[0],
            prediction.bbox[1] > 10 ? prediction.bbox[1] - 5 : 10
        );
    });

    // Call detectObjects again after detectionFrequency ms
    setTimeout(detectObjects, detectionFrequency);
}

// Optimize TensorFlow.js backend
async function optimizeBackend() {
    const backends = await tf.engine().getSortedBackends();
    if (backends.includes("webgpu")) {
        await tf.setBackend("webgpu"); // Use WebGPU for cutting-edge performance
        console.log("WebGPU backend enabled!");
    } else {
        await tf.setBackend("webgl"); // Fallback to WebGL
        console.log("WebGL backend enabled!");
    }
}

// Run everything when the page loads
window.onload = async function () {
    await optimizeBackend();
    await startWebcam();
    await loadModel();
};
