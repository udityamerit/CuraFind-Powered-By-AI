/* particlesJS.load(@dom-id, @path-json, @callback (optional)); */
particlesJS.load('particles-js', 'static/particles.json', function() {
  console.log('callback - particles.js config loaded');
});

document.addEventListener('DOMContentLoaded', function () {
    const searchForm = document.querySelector('#search-form');
    const searchInput = searchForm.querySelector('input[name="query"]');
    const micBtn = document.querySelector('#mic-btn');

    if ('webkitSpeechRecognition' in window) {
        const recognition = new webkitSpeechRecognition();
        recognition.continuous = false;
        recognition.interimResults = false;
        recognition.lang = 'en-US';

        micBtn.addEventListener('click', () => {
            recognition.start();
        });

        recognition.onresult = (event) => {
            const transcript = event.results[0][0].transcript;
            searchInput.value = transcript;
            searchForm.submit();
        };

        recognition.onerror = (event) => {
            console.error('Speech recognition error:', event.error);
        };

    } else {
        micBtn.style.display = 'none';
        console.log('Speech recognition not supported in this browser.');
    }
});