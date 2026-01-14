// 全局变量
let originalImage = null;
let currentTool = 'brush'; // 'brush' or 'rect'
let brushSize = 20;
let isDrawing = false;
let startX = 0;
let startY = 0;
let lastX = 0; // 上一次画笔位置（用于连续绘制）
let lastY = 0;
let maskDrawn = false;

// 图片查看器相关
let generatedImages = []; // 存储生成的图片数据
let currentImageIndex = 0;
let maskOverlayCanvas = null;
let maskOverlayCtx = null;
let showMaskOverlay = false;
let showCropOverlay = false;
let currentMaskImageData = null;
let currentCropInfo = null; // 裁剪区域信息
let currentBbox = null; // 掩码外接矩形信息

// Canvas元素和Context（延迟初始化）
let originalCanvas, maskCanvas, drawCanvas;
let originalCtx, maskCtx, drawCtx;
let canvasEventsInitialized = false; // 标记事件是否已初始化

// 初始化Canvas元素
function initCanvas() {
    originalCanvas = document.getElementById('original-canvas');
    maskCanvas = document.getElementById('mask-canvas');
    drawCanvas = document.getElementById('draw-canvas');
    
    if (originalCanvas && maskCanvas && drawCanvas) {
        originalCtx = originalCanvas.getContext('2d');
        maskCtx = maskCanvas.getContext('2d');
        drawCtx = drawCanvas.getContext('2d');
        
        // 初始化事件监听器
        if (!initCanvasEvents()) {
            console.error('Canvas事件监听器初始化失败');
            return false;
        }
        
        return true;
    }
    return false;
}

// 标签页切换
function switchTab(tabName, eventElement) {
    try {
        // 隐藏所有标签页
        document.querySelectorAll('.tab-content').forEach(tab => {
            tab.classList.remove('active');
        });
        
        // 移除所有按钮的active状态
        document.querySelectorAll('.tab-button').forEach(btn => {
            btn.classList.remove('active');
        });
        
        // 显示选中的标签页
        const tabContent = document.getElementById(tabName + '-tab');
        if (tabContent) {
            tabContent.classList.add('active');
        }
        
        // 设置按钮active状态
        if (eventElement) {
            eventElement.classList.add('active');
        } else {
            // 如果没有传入eventElement，通过tabName查找对应的按钮
            document.querySelectorAll('.tab-button').forEach(btn => {
                if (btn.getAttribute('onclick') && btn.getAttribute('onclick').includes(tabName)) {
                    btn.classList.add('active');
                }
            });
        }
    } catch (error) {
        console.error('切换标签页时出错:', error);
    }
}

// 加载模型
async function loadModels() {
    const loadBtn = document.getElementById('load-btn');
    const loadStatus = document.getElementById('load-status');
    
    loadBtn.disabled = true;
    loadStatus.textContent = '正在加载模型...';
    loadStatus.className = 'status-message info';
    
    const formData = new FormData();
    
    // 添加SD3路径或文件（优先使用路径）
    const sd3Path = document.getElementById('sd3_path').value.trim();
    if (sd3Path) {
        formData.append('sd3_path', sd3Path);
    } else {
        const sd3File = document.getElementById('sd3_file').files[0];
        if (sd3File) {
            formData.append('sd3_file', sd3File);
        }
    }
    
    // 添加LoRA路径或文件（优先使用路径）
    const loraPath = document.getElementById('lora_path').value.trim();
    if (loraPath) {
        formData.append('lora_path', loraPath);
    } else {
        const loraFile = document.getElementById('lora_file').files[0];
        if (loraFile) {
            formData.append('lora_file', loraFile);
        }
    }
    
    try {
        const response = await fetch('/api/load_models', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.success) {
            loadStatus.textContent = data.message;
            loadStatus.className = 'status-message success';
            
            // 启用生成按钮
            document.getElementById('generate-btn').disabled = false;
        } else {
            loadStatus.textContent = '错误: ' + data.message;
            loadStatus.className = 'status-message error';
        }
    } catch (error) {
        loadStatus.textContent = '错误: ' + error.message;
        loadStatus.className = 'status-message error';
    } finally {
        loadBtn.disabled = false;
    }
}

// 处理图片上传
function handleImageUpload(event) {
    const file = event.target.files[0];
    if (!file) return;
    
    const reader = new FileReader();
    reader.onload = function(e) {
        const img = new Image();
        img.onload = function() {
            originalImage = img;
            
            // 使用原始图片尺寸，不进行resize
            const width = img.width;
            const height = img.height;
            
            console.log(`图片尺寸: ${width} x ${height}`);
            
            // 确保Canvas已初始化
            if (!originalCanvas || !maskCanvas || !drawCanvas) {
                if (!initCanvas()) {
                    console.error('Canvas元素未找到');
                    return;
                }
            }
            
            // 确保事件监听器已初始化
            if (!canvasEventsInitialized) {
                console.log('重新初始化Canvas事件监听器');
                if (!initCanvasEvents()) {
                    console.error('Canvas事件监听器初始化失败');
                }
            }
            
            // 设置canvas的实际尺寸为原始分辨率（用于绘制和数据处理）
            originalCanvas.width = maskCanvas.width = drawCanvas.width = width;
            originalCanvas.height = maskCanvas.height = drawCanvas.height = height;
            
            // 计算显示尺寸（自适应，但保持宽高比）
            const maxDisplayWidth = 1200; // 最大显示宽度
            const maxDisplayHeight = 800;  // 最大显示高度
            let displayWidth = width;
            let displayHeight = height;
            
            // 如果图片太大，按比例缩小显示尺寸
            if (displayWidth > maxDisplayWidth) {
                displayHeight = (displayHeight * maxDisplayWidth) / displayWidth;
                displayWidth = maxDisplayWidth;
            }
            if (displayHeight > maxDisplayHeight) {
                displayWidth = (displayWidth * maxDisplayHeight) / displayHeight;
                displayHeight = maxDisplayHeight;
            }
            
            // 设置CSS显示尺寸（自适应显示，但canvas内部保持原始分辨率）
            originalCanvas.style.width = maskCanvas.style.width = drawCanvas.style.width = displayWidth + 'px';
            originalCanvas.style.height = maskCanvas.style.height = drawCanvas.style.height = displayHeight + 'px';
            
            // 设置canvas-container的尺寸，确保容器有正确的高度（因为canvas是absolute定位）
            const canvasContainer = document.querySelector('.canvas-container');
            if (canvasContainer) {
                canvasContainer.style.width = displayWidth + 'px';
                canvasContainer.style.height = displayHeight + 'px';
            }
            
            console.log(`Canvas实际尺寸: ${width} x ${height}, 显示尺寸: ${displayWidth} x ${displayHeight}`);
            
            // 绘制原始图片（使用原始尺寸）
            originalCtx.drawImage(img, 0, 0, width, height);
            
            // 初始化掩码canvas为透明（黑色背景，但canvas本身透明）
            maskCtx.fillStyle = 'black';
            maskCtx.fillRect(0, 0, width, height);
            // 设置mask-canvas为透明，这样可以看到下面的原始图片
            maskCanvas.style.opacity = '0';
            
            // 清除绘制canvas
            drawCtx.clearRect(0, 0, width, height);
            
            // 显示canvas容器
            document.getElementById('image-preview-container').classList.remove('hidden');
            
            maskDrawn = false;
        };
        img.src = e.target.result;
    };
    reader.readAsDataURL(file);
}

// 设置工具
function setTool(tool) {
    currentTool = tool;
    
    // 更新按钮状态
    document.getElementById('brush-btn').classList.toggle('active', tool === 'brush');
    document.getElementById('rect-btn').classList.toggle('active', tool === 'rect');
    
    // 更新光标
    drawCanvas.style.cursor = tool === 'brush' ? 'crosshair' : 'crosshair';
}

// 更新画笔大小
function updateBrushSize() {
    brushSize = parseInt(document.getElementById('brush-size').value);
    document.getElementById('brush-size-value').textContent = brushSize;
}

// 清除画布
function clearCanvas() {
    const width = maskCanvas.width;
    const height = maskCanvas.height;
    
    // 清除掩码canvas（恢复为黑色）
    maskCtx.fillStyle = 'black';
    maskCtx.fillRect(0, 0, width, height);
    maskCanvas.style.opacity = '0';
    
    // 清除绘制canvas
    drawCtx.clearRect(0, 0, width, height);
    
    maskDrawn = false;
    // 重置上一次的位置
    lastX = 0;
    lastY = 0;
    // 隐藏掩码信息
    document.getElementById('mask-info-section').style.display = 'none';
}

// 绘制函数 - 画笔（连续线条）
function drawBrush(x, y, isFirstPoint = false) {
    // 在绘制canvas上显示白色画笔（用于视觉反馈）
    drawCtx.fillStyle = 'white';
    drawCtx.strokeStyle = 'white';
    drawCtx.lineWidth = brushSize;
    drawCtx.lineCap = 'round'; // 圆形端点，使线条更平滑
    drawCtx.lineJoin = 'round'; // 圆形连接，使线条更平滑
    
    // 同时更新掩码canvas（实际掩码数据）
    maskCtx.fillStyle = 'white';
    maskCtx.strokeStyle = 'white';
    maskCtx.lineWidth = brushSize;
    maskCtx.lineCap = 'round';
    maskCtx.lineJoin = 'round';
    
    if (isFirstPoint) {
        // 第一个点，画一个圆
        drawCtx.beginPath();
        drawCtx.arc(x, y, brushSize / 2, 0, Math.PI * 2);
        drawCtx.fill();
        
        maskCtx.beginPath();
        maskCtx.arc(x, y, brushSize / 2, 0, Math.PI * 2);
        maskCtx.fill();
    } else {
        // 连续绘制，画线连接上一个点和当前点
        drawCtx.beginPath();
        drawCtx.moveTo(lastX, lastY);
        drawCtx.lineTo(x, y);
        drawCtx.stroke();
        
        maskCtx.beginPath();
        maskCtx.moveTo(lastX, lastY);
        maskCtx.lineTo(x, y);
        maskCtx.stroke();
    }
    
    // 更新上一次的位置
    lastX = x;
    lastY = y;
    
    // 显示掩码canvas（半透明，让用户看到绘制的区域）
    maskCanvas.style.opacity = '0.5';
    
    maskDrawn = true;
}

// 绘制矩形
function drawRect(x1, y1, x2, y2) {
    const width = Math.abs(x2 - x1);
    const height = Math.abs(y2 - y1);
    const startX = Math.min(x1, x2);
    const startY = Math.min(y1, y2);
    
    // 清除绘制canvas
    drawCtx.clearRect(0, 0, drawCanvas.width, drawCanvas.height);
    
    // 绘制矩形轮廓
    drawCtx.strokeStyle = 'white';
    drawCtx.lineWidth = 2;
    drawCtx.strokeRect(startX, startY, width, height);
}

// 初始化Canvas事件监听器
function initCanvasEvents() {
    if (!drawCanvas) {
        console.error('drawCanvas未初始化，无法绑定事件');
        return false;
    }
    
    // 如果已经初始化过，跳过
    if (canvasEventsInitialized) {
        return true;
    }
    
    // 绑定事件监听器
    drawCanvas.addEventListener('mousedown', (e) => {
        if (!originalImage) {
            console.log('原始图片未加载，无法绘制');
            return;
        }
        
        isDrawing = true;
        const rect = drawCanvas.getBoundingClientRect();
        const scaleX = drawCanvas.width / rect.width;
        const scaleY = drawCanvas.height / rect.height;
        
        startX = (e.clientX - rect.left) * scaleX;
        startY = (e.clientY - rect.top) * scaleY;
        
        // 初始化上一次的位置
        lastX = startX;
        lastY = startY;
        
        if (currentTool === 'brush') {
            drawBrush(startX, startY, true); // true表示第一个点
        }
    });

    drawCanvas.addEventListener('mousemove', (e) => {
        if (!isDrawing || !originalImage) return;
        
        const rect = drawCanvas.getBoundingClientRect();
        const scaleX = drawCanvas.width / rect.width;
        const scaleY = drawCanvas.height / rect.height;
        
        const x = (e.clientX - rect.left) * scaleX;
        const y = (e.clientY - rect.top) * scaleY;
        
        if (currentTool === 'brush') {
            drawBrush(x, y, false); // false表示连续绘制
        } else if (currentTool === 'rect') {
            drawRect(startX, startY, x, y);
        }
    });

    drawCanvas.addEventListener('mouseup', (e) => {
        if (!isDrawing || !originalImage) return;
        
        if (currentTool === 'rect') {
            const rect = drawCanvas.getBoundingClientRect();
            const scaleX = drawCanvas.width / rect.width;
            const scaleY = drawCanvas.height / rect.height;
            
            const endX = (e.clientX - rect.left) * scaleX;
            const endY = (e.clientY - rect.top) * scaleY;
            
            // 填充矩形到掩码canvas
            const width = Math.abs(endX - startX);
            const height = Math.abs(endY - startY);
            const startX2 = Math.min(startX, endX);
            const startY2 = Math.min(startY, endY);
            
            maskCtx.fillStyle = 'white';
            maskCtx.fillRect(startX2, startY2, width, height);
            
            // 显示掩码canvas（半透明）
            maskCanvas.style.opacity = '0.5';
            
            // 清除绘制canvas
            drawCtx.clearRect(0, 0, drawCanvas.width, drawCanvas.height);
            
            maskDrawn = true;
            // 延迟更新掩码信息，确保canvas已更新
            setTimeout(() => {
                updateMaskInfo();
            }, 100);
        }
        
        isDrawing = false;
        // 重置上一次的位置
        lastX = 0;
        lastY = 0;
    });

    drawCanvas.addEventListener('mouseleave', () => {
        if (currentTool === 'rect') {
            drawCtx.clearRect(0, 0, drawCanvas.width, drawCanvas.height);
        }
        isDrawing = false;
    });
    
    canvasEventsInitialized = true;
    console.log('Canvas事件监听器已绑定');
    return true;
}


// 计算自动padding
async function calculateAutoPadding() {
    if (!originalImage || !maskDrawn) {
        alert('请先上传图片并绘制掩码区域！');
        return;
    }
    
    const autoPaddingBtn = document.getElementById('auto-padding-btn');
    autoPaddingBtn.disabled = true;
    autoPaddingBtn.textContent = '计算中...';
    
    try {
        const originalImageData = originalCanvas.toDataURL('image/png');
        const maskImageData = maskCanvas.toDataURL('image/png');
        
        const response = await fetch('/api/calculate_mask_info', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                original_image: originalImageData,
                mask_image: maskImageData
            })
        });
        
        const result = await response.json();
        
        if (result.success) {
            // 设置自动计算的padding
            document.getElementById('padding_mask_crop').value = result.auto_padding;
            
            // 更新掩码信息显示
            updateMaskInfoDisplay(result.bbox, result.auto_padding);
        } else {
            alert('计算失败: ' + result.message);
        }
    } catch (error) {
        alert('计算出错: ' + error.message);
    } finally {
        autoPaddingBtn.disabled = false;
        autoPaddingBtn.textContent = '自动计算';
    }
}

// 更新掩码信息显示
function updateMaskInfo() {
    if (!maskDrawn) {
        document.getElementById('mask-info-section').style.display = 'none';
        return;
    }
    
    // 调用API计算掩码信息
    const originalImageData = originalCanvas.toDataURL('image/png');
    const maskImageData = maskCanvas.toDataURL('image/png');
    
    fetch('/api/calculate_mask_info', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            original_image: originalImageData,
            mask_image: maskImageData
        })
    })
    .then(response => response.json())
    .then(result => {
        if (result.success) {
            updateMaskInfoDisplay(result.bbox, result.auto_padding);
        }
    })
    .catch(error => {
        console.error('获取掩码信息失败:', error);
    });
}

// 更新掩码信息显示
function updateMaskInfoDisplay(bbox, autoPadding) {
    if (!bbox) {
        document.getElementById('mask-info-section').style.display = 'none';
        return;
    }
    
    document.getElementById('mask-info-section').style.display = 'block';
    document.getElementById('bbox-info').textContent = `位置: (${bbox.x}, ${bbox.y}), 尺寸: ${bbox.width} x ${bbox.height}`;
    document.getElementById('model-size-info').textContent = '512 x 512';
}

// 生成缺陷
async function generateDefects() {
    if (!originalImage) {
        alert('请先上传图片！');
        return;
    }
    
    if (!maskDrawn) {
        alert('请先在图片上绘制掩码区域！');
        return;
    }
    
    const generateBtn = document.getElementById('generate-btn');
    const generateStatus = document.getElementById('generate-status');
    
    generateBtn.disabled = true;
    generateStatus.textContent = '正在生成...这可能需要一些时间，请耐心等待';
    generateStatus.className = 'status-message info';
    
    try {
        // 获取原始图片和掩码的base64
        const originalImageData = originalCanvas.toDataURL('image/png');
        const maskImageData = maskCanvas.toDataURL('image/png');
        
        // 获取参数
        
        const paddingMaskCropValue = document.getElementById('padding_mask_crop').value;
        const paddingMaskCrop = paddingMaskCropValue ? parseInt(paddingMaskCropValue) : null;
        
        const data = {
            original_image: originalImageData,
            mask_image: maskImageData,
            prompt: document.getElementById('prompt').value || 'defect of crack',
            negative_prompt: document.getElementById('negative_prompt').value,
            num_images: parseInt(document.getElementById('num_images').value) || 4,
            guidance_scale: parseFloat(document.getElementById('guidance_scale').value) || 7.0,
            num_inference_steps: parseInt(document.getElementById('num_inference_steps').value) || 28,
            padding_mask_crop: paddingMaskCrop
        };
        
        const response = await fetch('/api/generate', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        });
        
        const result = await response.json();
        
        if (result.success) {
            generateStatus.textContent = result.message + ' (保存路径: ' + result.output_dir + ')';
            generateStatus.className = 'status-message success';
            
            // 保存生成的图片数据
            generatedImages = result.images;
            currentImageIndex = 0;
            
            // 保存当前掩码数据用于显示
            if (maskCanvas) {
                currentMaskImageData = maskCanvas.toDataURL('image/png');
            }
            
            // 保存裁剪区域和边界框信息
            currentCropInfo = result.crop_info || null;
            currentBbox = result.bbox || null;
            
            // 初始化图片查看器
            initializeImageViewer();
            
            // 显示第一张图片
            displayImage(0);
            
            // 滚动到结果区域
            document.querySelector('.results-section').scrollIntoView({ behavior: 'smooth' });
        } else {
            generateStatus.textContent = '错误: ' + result.message;
            generateStatus.className = 'status-message error';
        }
    } catch (error) {
        generateStatus.textContent = '错误: ' + error.message;
        generateStatus.className = 'status-message error';
    } finally {
        generateBtn.disabled = false;
    }
}

// 初始化图片查看器
function initializeImageViewer() {
    // 隐藏placeholder，显示查看器
    document.getElementById('image-viewer-placeholder').classList.add('hidden');
    document.getElementById('image-viewer-content').classList.remove('hidden');
    
    // 初始化掩码overlay canvas
    const maskOverlay = document.getElementById('mask-overlay');
    if (!maskOverlayCanvas) {
        maskOverlayCanvas = document.createElement('canvas');
        maskOverlay.appendChild(maskOverlayCanvas);
        maskOverlayCtx = maskOverlayCanvas.getContext('2d');
    }
    
    // 启用控制按钮
    document.getElementById('prev-btn').disabled = false;
    document.getElementById('next-btn').disabled = false;
    document.getElementById('show-mask-btn').disabled = false;
    document.getElementById('show-crop-btn').disabled = false;
    document.getElementById('download-all-btn').disabled = false;
    
    // 更新计数器
    updateImageCounter();
}

// 显示指定索引的图片
function displayImage(index) {
    if (generatedImages.length === 0 || index < 0 || index >= generatedImages.length) {
        return;
    }
    
    currentImageIndex = index;
    const viewerImage = document.getElementById('viewer-image');
    
    // 设置新的图片源
    viewerImage.src = generatedImages[index];
    
    // 更新计数器
    updateImageCounter();
    
    // 更新按钮状态
    document.getElementById('prev-btn').disabled = (index === 0);
    document.getElementById('next-btn').disabled = (index === generatedImages.length - 1);
    
    // 图片加载完成后，如果显示掩码或裁剪区域，更新overlay
    viewerImage.onload = () => {
        if (showMaskOverlay && currentMaskImageData) {
            // 延迟一下确保图片尺寸已经更新
            setTimeout(() => {
                updateMaskOverlay();
            }, 100);
        }
        if (showCropOverlay && currentCropInfo) {
            setTimeout(() => {
                updateCropOverlay();
            }, 100);
        }
    };
}

// 更新图片计数器
function updateImageCounter() {
    const counter = document.getElementById('image-counter');
    counter.textContent = `${currentImageIndex + 1} / ${generatedImages.length}`;
}

// 上一张图片
function previousImage() {
    if (currentImageIndex > 0) {
        displayImage(currentImageIndex - 1);
    }
}

// 下一张图片
function nextImage() {
    if (currentImageIndex < generatedImages.length - 1) {
        displayImage(currentImageIndex + 1);
    }
}

// 切换掩码显示
function toggleMask() {
    showMaskOverlay = !showMaskOverlay;
    const maskOverlay = document.getElementById('mask-overlay');
    const showMaskText = document.getElementById('show-mask-text');
    
    if (showMaskOverlay) {
        maskOverlay.classList.remove('hidden');
        showMaskText.textContent = '隐藏掩码区域';
        // 如果同时显示掩码和裁剪区域，需要同时更新
        updateMaskOverlay();
        if (showCropOverlay && currentCropInfo) {
            updateCropOverlay();
        }
    } else {
        showMaskText.textContent = '显示掩码区域';
        // 如果裁剪区域也不显示，隐藏overlay
        if (!showCropOverlay) {
            maskOverlay.classList.add('hidden');
        } else {
            // 如果裁剪区域还在显示，只更新裁剪overlay
            updateCropOverlay();
        }
    }
}

// 切换裁剪区域显示
function toggleCrop() {
    showCropOverlay = !showCropOverlay;
    const maskOverlay = document.getElementById('mask-overlay');
    const showCropText = document.getElementById('show-crop-text');
    
    if (showCropOverlay) {
        maskOverlay.classList.remove('hidden');
        showCropText.textContent = '隐藏裁剪区域';
        // 如果同时显示掩码和裁剪区域，需要同时更新
        if (showMaskOverlay && currentMaskImageData) {
            updateMaskOverlay();
        }
        if (currentCropInfo) {
            updateCropOverlay();
        }
    } else {
        showCropText.textContent = '显示裁剪区域';
        // 如果掩码也不显示，隐藏overlay
        if (!showMaskOverlay) {
            maskOverlay.classList.add('hidden');
        } else {
            // 如果掩码还在显示，只更新掩码overlay
            updateMaskOverlay();
        }
    }
}

// 更新掩码overlay
function updateMaskOverlay() {
    if (!maskOverlayCanvas || !maskOverlayCtx || !currentMaskImageData) {
        return;
    }
    
    const viewerImage = document.getElementById('viewer-image');
    const imageViewer = document.querySelector('.image-viewer-content');
    
    // 等待图片加载完成
    const updateOverlay = () => {
        // 设置canvas尺寸与显示区域匹配
        const rect = imageViewer.getBoundingClientRect();
        maskOverlayCanvas.width = rect.width;
        maskOverlayCanvas.height = rect.height;
        
        // 获取图片的显示尺寸
        const imgRect = viewerImage.getBoundingClientRect();
        const imgLeft = (rect.width - imgRect.width) / 2;
        const imgTop = (rect.height - imgRect.height) / 2;
        
        // 加载掩码图片
        const maskImg = new Image();
        maskImg.onload = () => {
            // 清空canvas
            maskOverlayCtx.clearRect(0, 0, maskOverlayCanvas.width, maskOverlayCanvas.height);
            
            // 创建临时canvas来处理掩码
            const tempCanvas = document.createElement('canvas');
            tempCanvas.width = maskImg.width;
            tempCanvas.height = maskImg.height;
            const tempCtx = tempCanvas.getContext('2d');
            
            // 绘制掩码到临时canvas
            tempCtx.drawImage(maskImg, 0, 0);
            const maskData = tempCtx.getImageData(0, 0, tempCanvas.width, tempCanvas.height);
            const pixels = maskData.data;
            
            // 创建掩码图像数据（白色区域=要inpaint的区域，用半透明红色标记）
            for (let i = 0; i < pixels.length; i += 4) {
                const r = pixels[i];
                const g = pixels[i + 1];
                const b = pixels[i + 2];
                // 如果像素是白色（掩码区域），标记为半透明红色
                if (r > 200 && g > 200 && b > 200) {
                    pixels[i] = 255;     // R
                    pixels[i + 1] = 0;   // G
                    pixels[i + 2] = 0;   // B
                    pixels[i + 3] = 150; // A (半透明)
                } else {
                    // 非掩码区域透明
                    pixels[i + 3] = 0;
                }
            }
            
            tempCtx.putImageData(maskData, 0, 0);
            
            // 在整个显示区域绘制半透明暗色遮罩
            maskOverlayCtx.fillStyle = 'rgba(0, 0, 0, 0.5)';
            maskOverlayCtx.fillRect(0, 0, maskOverlayCanvas.width, maskOverlayCanvas.height);
            
            // 计算缩放比例
            const scaleX = imgRect.width / maskImg.width;
            const scaleY = imgRect.height / maskImg.height;
            
            // 在图片位置绘制掩码标记（红色区域表示要inpaint的区域）
            maskOverlayCtx.drawImage(
                tempCanvas,
                imgLeft, imgTop,
                maskImg.width * scaleX,
                maskImg.height * scaleY
            );
        };
        maskImg.src = currentMaskImageData;
    };
    
    if (viewerImage.complete) {
        updateOverlay();
    } else {
        viewerImage.onload = updateOverlay;
    }
}

// 更新裁剪区域overlay
function updateCropOverlay() {
    if (!maskOverlayCanvas || !maskOverlayCtx || !currentCropInfo) {
        return;
    }
    
    const viewerImage = document.getElementById('viewer-image');
    const imageViewer = document.querySelector('.image-viewer-content');
    
    // 等待图片加载完成
    const updateOverlay = () => {
        // 设置canvas尺寸与显示区域匹配
        const rect = imageViewer.getBoundingClientRect();
        maskOverlayCanvas.width = rect.width;
        maskOverlayCanvas.height = rect.height;
        
        // 获取图片的显示尺寸
        const imgRect = viewerImage.getBoundingClientRect();
        const imgLeft = (rect.width - imgRect.width) / 2;
        const imgTop = (rect.height - imgRect.height) / 2;
        
        // 获取原始图片的尺寸（生成的图片尺寸应该和原图一样）
        const originalWidth = originalCanvas ? originalCanvas.width : viewerImage.naturalWidth;
        const originalHeight = originalCanvas ? originalCanvas.height : viewerImage.naturalHeight;
        
        // 计算缩放比例
        const scaleX = imgRect.width / originalWidth;
        const scaleY = imgRect.height / originalHeight;
        
        // 计算裁剪区域在显示图片中的位置和尺寸
        const cropX = currentCropInfo.x * scaleX + imgLeft;
        const cropY = currentCropInfo.y * scaleY + imgTop;
        const cropWidth = currentCropInfo.width * scaleX;
        const cropHeight = currentCropInfo.height * scaleY;
        
        // 如果同时显示掩码，先绘制掩码overlay
        if (showMaskOverlay && currentMaskImageData) {
            // 先绘制掩码overlay的基础（暗色遮罩）
            maskOverlayCtx.fillStyle = 'rgba(0, 0, 0, 0.5)';
            maskOverlayCtx.fillRect(0, 0, maskOverlayCanvas.width, maskOverlayCanvas.height);
            
            // 然后绘制掩码区域（红色标记）
            // 这里需要加载掩码图片并绘制
            const maskImg = new Image();
            maskImg.onload = () => {
                const maskScaleX = imgRect.width / maskImg.width;
                const maskScaleY = imgRect.height / maskImg.height;
                
                // 创建临时canvas处理掩码
                const tempCanvas = document.createElement('canvas');
                tempCanvas.width = maskImg.width;
                tempCanvas.height = maskImg.height;
                const tempCtx = tempCanvas.getContext('2d');
                tempCtx.drawImage(maskImg, 0, 0);
                const maskData = tempCtx.getImageData(0, 0, tempCanvas.width, tempCanvas.height);
                const pixels = maskData.data;
                
                // 创建掩码图像数据（白色区域=要inpaint的区域，用半透明红色标记）
                for (let i = 0; i < pixels.length; i += 4) {
                    const r = pixels[i];
                    const g = pixels[i + 1];
                    const b = pixels[i + 2];
                    if (r > 200 && g > 200 && b > 200) {
                        pixels[i] = 255;
                        pixels[i + 1] = 0;
                        pixels[i + 2] = 0;
                        pixels[i + 3] = 150;
                    } else {
                        pixels[i + 3] = 0;
                    }
                }
                tempCtx.putImageData(maskData, 0, 0);
                
                // 绘制掩码标记
                maskOverlayCtx.drawImage(
                    tempCanvas,
                    imgLeft, imgTop,
                    maskImg.width * maskScaleX,
                    maskImg.height * maskScaleY
                );
                
                // 然后绘制裁剪区域边框
                maskOverlayCtx.strokeStyle = 'rgba(0, 150, 255, 0.8)';
                maskOverlayCtx.lineWidth = 3;
                maskOverlayCtx.strokeRect(cropX, cropY, cropWidth, cropHeight);
            };
            maskImg.src = currentMaskImageData;
        } else {
            // 只显示裁剪区域
            // 在整个显示区域绘制半透明暗色遮罩
            maskOverlayCtx.fillStyle = 'rgba(0, 0, 0, 0.5)';
            maskOverlayCtx.fillRect(0, 0, maskOverlayCanvas.width, maskOverlayCanvas.height);
            
            // 清除裁剪区域内的遮罩（显示裁剪区域）
            maskOverlayCtx.clearRect(cropX, cropY, cropWidth, cropHeight);
            
            // 绘制裁剪区域的边框（蓝色）
            maskOverlayCtx.strokeStyle = 'rgba(0, 150, 255, 0.8)';
            maskOverlayCtx.lineWidth = 3;
            maskOverlayCtx.strokeRect(cropX, cropY, cropWidth, cropHeight);
        }
    };
    
    if (viewerImage.complete) {
        updateOverlay();
    } else {
        viewerImage.onload = updateOverlay;
    }
}

// 下载所有图片
function downloadAllImages() {
    if (generatedImages.length === 0) {
        return;
    }
    
    // 为每张图片创建下载链接
    generatedImages.forEach((imgData, index) => {
        const link = document.createElement('a');
        link.href = imgData;
        link.download = `sd3_inpaint_${currentImageIndex}_${index + 1}.png`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        
        // 添加小延迟以避免浏览器阻止多个下载
        setTimeout(() => {}, 100 * index);
    });
}

// 键盘快捷键支持
document.addEventListener('keydown', (e) => {
    // 只在图片查看器激活时响应
    if (generatedImages.length === 0) return;
    
    if (e.key === 'ArrowLeft') {
        previousImage();
    } else if (e.key === 'ArrowRight') {
        nextImage();
    }
});

// 初始化
document.addEventListener('DOMContentLoaded', () => {
    try {
        console.log('页面加载完成，开始初始化...');
        
        // 初始化Canvas
        if (!initCanvas()) {
            console.error('Canvas元素未找到，请检查HTML结构');
        } else {
            console.log('Canvas初始化成功');
        }
        
        // 检查模型是否已加载
        fetch('/api/check_model')
            .then(response => {
                if (!response.ok) {
                    throw new Error('网络响应错误: ' + response.status);
                }
                return response.json();
            })
            .then(data => {
                console.log('模型状态检查完成:', data);
                const generateBtn = document.getElementById('generate-btn');
                if (generateBtn && data.loaded) {
                    generateBtn.disabled = false;
                    console.log('模型已加载，生成按钮已启用');
                }
            })
            .catch(error => {
                console.error('检查模型状态时出错:', error);
            });
        
        console.log('初始化完成');
    } catch (error) {
        console.error('初始化过程中发生错误:', error);
        alert('页面初始化失败，请刷新页面重试。错误信息: ' + error.message);
    }
});

