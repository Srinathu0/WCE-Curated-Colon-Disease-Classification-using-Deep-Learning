// document.addEventListener('DOMContentLoaded', function () {
//     const navToggle = document.querySelector('.nav-toggle');
//     const navLinks = document.querySelector('.nav-links');

//     navToggle.addEventListener('click', function () {
//         navLinks.classList.toggle('open');
//     });

//     const uploadForm = document.getElementById('upload-form');
//     uploadForm.addEventListener('submit', function (event) {
//         event.preventDefault();
//         const fileInput = uploadForm.querySelector('input[type="file"]');
//         const resultDiv = document.getElementById('result');
//         const predictionSpan = document.getElementById('prediction');

//         if (fileInput.files.length === 0) {
//             alert('Please select a file.');
//             return;
//         }

//         const formData = new FormData();
//         formData.append('file', fileInput.files[0]);

//         fetch('/predict', {
//             method: 'POST',
//             body: formData
//         })
//         .then(response => response.json())
//         .then(data => {
//             if (data.success) {
//                 predictionSpan.textContent = data.prediction;
//                 resultDiv.style.display = 'block';
//             } else {
//                 alert('Prediction failed.');
//             }
//         })
//         .catch(error => {
//             console.error('Error:', error);
//             alert('An error occurred while uploading the file.');
//         });
//     });
// });

// document.addEventListener('DOMContentLoaded', function () {
//     const navToggle = document.querySelector('.nav-toggle');
//     const navLinks = document.querySelector('.nav-links');

//     navToggle.addEventListener('click', function () {
//         navLinks.classList.toggle('open');
//     });

//     const uploadForm = document.getElementById('upload-form');
//     uploadForm.addEventListener('submit', function () {
//         const resultDiv = document.getElementById('result');
//         resultDiv.style.display = 'none';
//     });
// });

// window.onload = function() {
//     const resultDiv = document.getElementById('result');
//     const prediction = document.getElementById('prediction');
//     if (prediction && prediction.innerText.trim() !== '') {
//         resultDiv.style.display = 'block';
//     }
// };


// new
// script.js

// Responsive navigation menu toggle
document.addEventListener('DOMContentLoaded', function() {
    const navToggle = document.querySelector('.nav-toggle');
    const navLinks = document.querySelector('.nav-links');

    navToggle.addEventListener('click', function() {
        navLinks.classList.toggle('open');
    });
});

// File input preview
document.addEventListener('DOMContentLoaded', function() {
    const fileInput = document.querySelector('input[type="file"]');
    const preview = document.createElement('img');
    preview.style.maxWidth = '100%';
    preview.style.marginTop = '10px';

    fileInput.addEventListener('change', function(event) {
        const file = event.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = function(e) {
                preview.src = e.target.result;
                fileInput.insertAdjacentElement('afterend', preview);
            };
            reader.readAsDataURL(file);
        }
    });
});
