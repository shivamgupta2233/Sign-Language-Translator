setInterval(async () => {

    const res = await fetch("/prediction");

    const data = await res.json();

    document.getElementById("label").innerHTML = data.label;

    document.getElementById("confidence").innerHTML =
        data.confidence + "%";

    document.getElementById("word").innerHTML =
        data.word;

    document.getElementById("sentence").innerHTML =
        data.sentence;

},200);