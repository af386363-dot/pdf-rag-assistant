async function uploadFile() {
    const fileInput = document.getElementById("pdf-file");
    const status = document.getElementById("status");
    const uploadBtn = document.getElementById("upload-btn");
    const questionInput = document.getElementById("question");
    const askBtn = document.getElementById("ask-btn");

    if (fileInput.files.length === 0) {
        status.style.color = "#e57373";
        status.innerText = "⚠️ Please choose a file first.";
        return;
    }

    const fileName = fileInput.files[0].name;

    uploadBtn.disabled = true;
    uploadBtn.innerText = "Uploading...";
    status.style.color = "#d4af37";
    status.innerText = `⏳ Uploading "${fileName}"...`;

    const formData = new FormData();
    formData.append("file", fileInput.files[0]);

    try {
        const response = await fetch("/upload", {
            method: "POST",
            body: formData
        });

        const data = await response.json();

        status.style.color = "#7ccf83";
        status.innerText = `✅ File uploaded — you can now ask questions.`;

        questionInput.disabled = false;
        questionInput.placeholder = "What would you like to know?";
        askBtn.disabled = false;

    } catch (error) {
        status.style.color = "#e57373";
        status.innerText = "❌ Upload failed. Check the console and try again.";
    } finally {
        uploadBtn.disabled = false;
        uploadBtn.innerText = "Upload & Analyze";
    }
}

async function askQuestion() {
    const questionInput = document.getElementById("question");
    const question = questionInput.value;

    if (question.trim() === "") {
        return;
    }

    const answerBox = document.getElementById("answer-box");
    const answerText = document.getElementById("answer-text");
    const detailsText = document.getElementById("details-text");
    const extraDetails = document.getElementById("extra-details");
    const resolvedQueryText = document.getElementById("resolved-query-text");
    const hydeText = document.getElementById("hyde-text");
    const chunksList = document.getElementById("chunks-list");

    answerText.innerText = "Thinking...";
    answerBox.style.display = "block";
    detailsText.innerText = "";
    extraDetails.style.display = "none";

    try {
        const response = await fetch("/ask", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ question: question })
        });

        const data = await response.json();

        answerText.innerText = data.answer;

        if (data.used_retrieval) {
            detailsText.innerText = `Faithfulness check: ${data.faithfulness}`;

            resolvedQueryText.innerText = data.resolved_query;
            hydeText.innerText = data.hypothetical_answer;

            chunksList.innerHTML = "";
            data.top_chunks.forEach(function (chunk, index) {
                const chunkDiv = document.createElement("div");
                chunkDiv.className = "chunk";
                chunkDiv.innerText = `Chunk ${index + 1}: ${chunk}`;
                chunksList.appendChild(chunkDiv);
            });

            extraDetails.style.display = "block";
        } else {
            detailsText.innerText = "No document retrieval was needed for this message.";
            extraDetails.style.display = "none";
        }
    } catch (error) {
        answerText.innerText = "Something went wrong while getting the answer.";
    }
}