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
}