// ===================================
// Global Variables
// ===================================

let lastSpoken = "";
let speaking = false;

// ===================================
// Update Prediction
// ===================================

async function updatePrediction() {

    try {

        const response = await fetch("/prediction");
        const data = await response.json();

        const label = (data.label || "").trim();
        const confidence = data.confidence || 0;
        const word = data.word || "";
        const sentence = data.sentence || "";

        document.getElementById("label").innerHTML = label;
        document.getElementById("confidence").innerHTML = confidence + "%";
        document.getElementById("word").innerHTML = word;
        document.getElementById("sentence").innerHTML = sentence;

        document.getElementById("progress").style.width = confidence + "%";

        // ==========================
        // Auto Speak
        // ==========================

        if (
            label !== "" &&
            label.toLowerCase() !== "unknown" &&
            label.toLowerCase() !== "waiting..." &&
            label.toLowerCase() !== "no hand" &&
            label !== lastSpoken &&
            !speaking
        ) {

            speaking = true;

            window.speechSynthesis.cancel();

            const speech = new SpeechSynthesisUtterance(label);

            speech.lang = "en-US";
            speech.rate = 0.9;
            speech.pitch = 1;
            speech.volume = 1;

            speech.onend = () => {
                speaking = false;
            };

            window.speechSynthesis.speak(speech);

            lastSpoken = label;
        }

    } catch (err) {

        console.error(err);

    }

}

// Refresh every 500 ms
setInterval(updatePrediction, 500);

// ===================================
// Speak Button
// ===================================

document.getElementById("speakBtn").onclick = async () => {

    const res = await fetch("/prediction");
    const data = await res.json();

    let text = data.word;

    if (text === "")
        text = data.label;

    if (
        text.toLowerCase() === "unknown" ||
        text.toLowerCase() === "waiting..."
    ) return;

    window.speechSynthesis.cancel();

    const speech = new SpeechSynthesisUtterance(text);

    speech.lang = "en-US";
    speech.rate = 0.9;
    speech.pitch = 1;
    speech.volume = 1;

    window.speechSynthesis.speak(speech);

};

// ===================================
// Clear Button
// ===================================

document.getElementById("clearBtn").onclick = async () => {

    await fetch("/clear");

    document.getElementById("word").innerHTML = "";
    document.getElementById("sentence").innerHTML = "";

};

// ===================================
// Save Button
// ===================================

document.getElementById("saveBtn").onclick = async () => {

    await fetch("/save");

    alert("History Saved Successfully");

};