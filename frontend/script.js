const canvas = document.getElementById('drawing-board');
const ctx = canvas.getContext('2d');
const clearBtn = document.getElementById('clear-btn');
const predictBtn = document.getElementById('predict-btn');
const predText = document.getElementById('prediction-text');
const confText = document.getElementById('confidence-text');

// 1. 初始化画板
ctx.lineWidth = 15; // 笔触要粗一点，因为最后会被缩放到 28x28
ctx.lineCap = 'round';
ctx.strokeStyle = 'black'; // 黑笔
ctx.fillStyle = 'white';   // 确保背景是纯白
ctx.fillRect(0, 0, canvas.width, canvas.height);

let isDrawing = false;

// 2. 绘图事件监听 (鼠标/触控)
function startDrawing(e) {
    isDrawing = true;
    draw(e);
}

function stopDrawing() {
    isDrawing = false;
    ctx.beginPath(); // 结束路径，防止连笔
}

function draw(e) {
    if (!isDrawing) return;

    // 获取鼠标在 Canvas 内的坐标
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    ctx.lineTo(x, y);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(x, y);
}

canvas.addEventListener('mousedown', startDrawing);
canvas.addEventListener('mouseup', stopDrawing);
canvas.addEventListener('mousemove', draw);
canvas.addEventListener('mouseout', stopDrawing);

// 3. 清除功能
clearBtn.addEventListener('click', () => {
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    predText.innerText = '?';
    confText.innerText = '0.00';
});

// 4. 识别功能 (核心网络请求)
predictBtn.addEventListener('click', () => {
    canvas.toBlob((blob) => {
        const formData = new FormData();
        formData.append('file', blob, 'drawing.png');

        predText.innerText = '识别中...';

        fetch('http://localhost:5000/predict', {
            method: 'POST',
            body: formData
        })
            .then(response => response.json())
            .then(data => {
                console.log("后端返回:", data);
                if (data.status === 'success') {
                    predText.innerText = data.prediction;
                    confText.innerText = (data.confidence * 100).toFixed(2) + '%';
                } else {
                    alert('识别失败: ' + data.error);
                }
            })
            .catch(error => {
                console.error('Error:', error);
                predText.innerText = '错误';
            });
    });
});