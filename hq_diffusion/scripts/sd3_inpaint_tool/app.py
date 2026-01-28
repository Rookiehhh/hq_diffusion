"""
SD3 Inpainting Flask Web应用
支持加载SD3模型、LoRA权重
支持在图片上绘制掩码进行缺陷生成
"""

import os
import sys
import torch
import numpy as np
from flask import Flask, render_template, request, jsonify, send_from_directory
from PIL import Image
import io
import base64
import diffusers
from diffusers.pipelines import StableDiffusion3InpaintPipeline
from diffusers.utils import numpy_to_pil
from datetime import datetime
import traceback
import gc
import PIL.Image

# 获取应用根目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__, 
            template_folder=os.path.join(BASE_DIR, 'templates'),
            static_folder=os.path.join(BASE_DIR, 'static'))
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, 'uploads')
app.config['OUTPUT_FOLDER'] = '/root/autodl-tmp/tmpimgs'

# 确保上传和输出文件夹存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# 全局变量
pipe = None
device = "cuda" if torch.cuda.is_available() else "cpu"
use_cpu_offload = False  # 是否启用CPU offload以节省显存
MODEL_INPUT_SIZE = 512  # 模型输入图像尺寸



def release_model_memory():
    """释放模型占用的显存"""
    global pipe
    
    print("正在释放模型显存...")
    
    # 删除pipeline
    if pipe is not None:
        try:
            # 将模型移到CPU
            if hasattr(pipe, 'to'):
                pipe.to('cpu')
            
            # 删除各个组件
            if hasattr(pipe, 'transformer'):
                del pipe.transformer
            if hasattr(pipe, 'vae'):
                del pipe.vae
            if hasattr(pipe, 'text_encoder'):
                del pipe.text_encoder
            if hasattr(pipe, 'text_encoder_2'):
                del pipe.text_encoder_2
            if hasattr(pipe, 'text_encoder_3'):
                del pipe.text_encoder_3
            if hasattr(pipe, 'tokenizer'):
                del pipe.tokenizer
            if hasattr(pipe, 'tokenizer_2'):
                del pipe.tokenizer_2
            if hasattr(pipe, 'tokenizer_3'):
                del pipe.tokenizer_3
            
            del pipe
            pipe = None
        except Exception as e:
            print(f"释放pipeline时出错: {e}")
    
    # 清理GPU缓存
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    
    # 强制垃圾回收
    gc.collect()
    
    print("显存释放完成")

def process_mask_from_base64(original_image, mask_base64):
    """从base64编码的掩码图像中提取掩码"""
    try:
        # 解码base64图像
        mask_data = base64.b64decode(mask_base64.split(',')[1])
        mask_image = Image.open(io.BytesIO(mask_data)).convert("RGB")
        
        # 确保尺寸匹配
        if mask_image.size != original_image.size:
            mask_image = mask_image.resize(original_image.size, Image.Resampling.LANCZOS)
        
        # 将RGB掩码转换为单通道掩码（白色区域=要inpaint的区域）
        mask_array = np.array(mask_image)
        # 将非黑色区域转换为白色（掩码）
        mask_gray = np.mean(mask_array, axis=2)
        mask = (mask_gray > 10).astype(np.uint8) * 255
        
        # 转换为单通道灰度图（Inpaint Pipeline 使用单通道掩码）
        mask_image = Image.fromarray(mask, mode='L')
        
        return mask_image
    except Exception as e:
        print(f"处理掩码时出错: {str(e)}")
        traceback.print_exc()
        return None

def calculate_mask_bbox(mask_image):
    """计算掩码的外接矩形（bounding box）
    
    Returns:
        tuple: (x, y, width, height) 或 None（如果没有掩码）
    """
    try:
        mask_array = np.array(mask_image)
        # 找到所有非零像素的位置
        coords = np.column_stack(np.where(mask_array > 10))
        
        if len(coords) == 0:
            return None
        
        # 计算边界框
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        
        x = int(x_min)
        y = int(y_min)
        width = int(x_max - x_min + 1)
        height = int(y_max - y_min + 1)
        
        return (x, y, width, height)
    except Exception as e:
        print(f"计算掩码边界框时出错: {str(e)}")
        return None

def calculate_auto_padding(bbox, model_size=512):
    """根据掩码外接矩形的最长边计算padding大小
    
    Args:
        bbox: (x, y, width, height) 掩码外接矩形
        model_size: 模型输入尺寸（默认512）
    
    Returns:
        int: padding大小，使得裁剪后的区域最长边等于model_size
    """
    if bbox is None:
        return 0
    
    x, y, width, height = bbox
    max_side = max(width, height)
    
    if max_side >= model_size:
        # 如果最长边已经大于等于模型尺寸，不需要padding
        return 0
    
    # 计算需要的padding，使得最长边等于model_size
    padding = (model_size - max_side) // 2
    
    return padding

@app.route('/')
def index():
    """主页面"""
    return render_template('index.html')

@app.route('/test')
def test():
    """测试页面"""
    return send_from_directory(os.path.dirname(__file__), 'test_page.html')

@app.route('/debug')
def debug():
    """调试信息页面"""
    import socket
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    
    info = {
        'hostname': hostname,
        'local_ip': local_ip,
        'flask_host': '0.0.0.0',
        'flask_port': 5000,
        'model_loaded': pipe is not None,
        'static_folder': app.static_folder,
        'template_folder': app.template_folder,
    }
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>调试信息</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1 {{ color: #333; }}
            .info {{ background: #f5f5f5; padding: 15px; margin: 10px 0; border-radius: 5px; }}
            .success {{ color: green; }}
            .error {{ color: red; }}
            .url {{ background: #e3f2fd; padding: 10px; margin: 10px 0; border-left: 4px solid #2196F3; }}
        </style>
    </head>
    <body>
        <h1>Flask应用调试信息</h1>
        <div class="info">
            <h2>访问地址：</h2>
            <div class="url">
                <strong>本地访问：</strong><br>
                <a href="http://127.0.0.1:5000/">http://127.0.0.1:5000/</a><br>
                <a href="http://localhost:5000/">http://localhost:5000/</a>
            </div>
            <div class="url">
                <strong>网络访问：</strong><br>
                <a href="http://{local_ip}:5000/">http://{local_ip}:5000/</a><br>
                <a href="http://172.17.0.7:5000/">http://172.17.0.7:5000/</a>
            </div>
        </div>
        <div class="info">
            <h2>应用状态：</h2>
            <p>模型已加载: <span class="{'success' if info['model_loaded'] else 'error'}">{'是' if info['model_loaded'] else '否'}</span></p>
            <p>静态文件夹: {info['static_folder']}</p>
            <p>模板文件夹: {info['template_folder']}</p>
        </div>
        <div class="info">
            <h2>测试链接：</h2>
            <ul>
                <li><a href="/">主页面</a></li>
                <li><a href="/test">测试页面</a></li>
                <li><a href="/api/check_model">API: 检查模型状态</a></li>
                <li><a href="/static/js/main.js">静态文件: main.js</a></li>
                <li><a href="/static/css/style.css">静态文件: style.css</a></li>
            </ul>
        </div>
        <div class="info">
            <h2>JavaScript测试：</h2>
            <p id="js-status">正在检查...</p>
            <script>
                document.getElementById('js-status').innerHTML = '<span class="success">✓ JavaScript正常工作</span>';
                console.log('调试页面: JavaScript正常');
            </script>
        </div>
    </body>
    </html>
    """
    return html

@app.route('/api/load_models', methods=['POST'])
def load_models():
    """加载模型API"""
    global pipe
    
    try:
        # 在加载新模型之前，释放旧的模型和显存
        release_model_memory()
        data = request.form
        
        # 获取文件路径（优先使用路径输入，否则使用上传的文件）
        sd3_path = None
        lora_path = None
        
        # 处理SD3模型：优先使用路径输入
        if data.get('sd3_path') and data.get('sd3_path').strip():
            sd3_path = data.get('sd3_path').strip()
            if not os.path.exists(sd3_path):
                return jsonify({'success': False, 'message': f'SD3模型路径不存在: {sd3_path}'}), 400
        elif 'sd3_file' in request.files:
            file = request.files['sd3_file']
            if file.filename:
                filename = f"sd3_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                sd3_path = filepath
        
        # 处理LoRA：优先使用路径输入
        if data.get('lora_path') and data.get('lora_path').strip():
            lora_path = data.get('lora_path').strip()
            if not os.path.exists(lora_path):
                return jsonify({'success': False, 'message': f'LoRA路径不存在: {lora_path}'}), 400
        elif 'lora_file' in request.files:
            file = request.files['lora_file']
            if file.filename:
                filename = f"lora_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                lora_path = filepath
        
        if not sd3_path:
            return jsonify({'success': False, 'message': '请提供SD3模型路径或上传文件'}), 400
        
        # 加载SD3 Inpaint pipeline
        print(f"正在加载SD3 Inpaint pipeline: {sd3_path}")
        pipe = StableDiffusion3InpaintPipeline.from_single_file(
            sd3_path,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32
        )
        
        # 加载LoRA权重（如果提供）
        if lora_path and os.path.exists(lora_path):
            print(f"正在加载LoRA权重: {lora_path}")
            pipe.load_lora_weights(lora_path)
        
        # 将模型移动到GPU并设置精度
        if device == "cuda":
            pipe.text_encoder.to(torch.bfloat16)
            pipe.text_encoder_2.to(torch.bfloat16)
            pipe.text_encoder_3.to(torch.bfloat16)
        
        # 检查是否需要使用CPU offload（如果显存不足）
        global use_cpu_offload
        if device == "cuda":
            # 检查可用显存
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            if total_memory < 24:  # 如果GPU显存小于24GB，启用CPU offload
                try:
                    # 注意：enable_model_cpu_offload需要模型在CPU上
                    pipe.enable_model_cpu_offload()
                    use_cpu_offload = True
                    print("已启用 CPU offload 以节省显存")
                except Exception as e:
                    print(f"启用CPU offload失败: {e}，将使用GPU模式")
                    pipe = pipe.to(device)
                    use_cpu_offload = False
            else:
                pipe = pipe.to(device)
                use_cpu_offload = False
            
            # 启用attention slicing以减少显存使用
            try:
                pipe.enable_attention_slicing()
                print("已启用 attention slicing")
            except:
                pass
            
            # 启用VAE slicing以减少VAE显存使用
            try:
                pipe.enable_vae_slicing()
                print("已启用 VAE slicing")
            except:
                pass
        else:
            pipe = pipe.to(device)
        
        # 清理一次缓存，确保显存使用正常
        memory_info = ""
        if device == "cuda":
            torch.cuda.empty_cache()
            # 显示显存使用情况
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            memory_info = f" (GPU显存: {allocated:.2f} GB)"
            print(f"GPU显存使用: 已分配 {allocated:.2f} GB, 已保留 {reserved:.2f} GB")
        
        return jsonify({
            'success': True,
            'message': f'模型加载成功！{memory_info}'
        })
    
    except Exception as e:
        error_msg = f"加载模型时出错: {str(e)}"
        print(error_msg)
        traceback.print_exc()
        return jsonify({
            'success': False,
            'message': error_msg
        }), 500

@app.route('/api/calculate_mask_info', methods=['POST'])
def calculate_mask_info():
    """计算掩码外接矩形和自动padding"""
    try:
        data = request.json
        original_image_base64 = data.get('original_image')
        mask_image_base64 = data.get('mask_image')
        
        if not original_image_base64 or not mask_image_base64:
            return jsonify({
                'success': False,
                'message': '请提供原始图片和掩码'
            }), 400
        
        # 解码原始图像
        original_data = base64.b64decode(original_image_base64.split(',')[1])
        original_image = Image.open(io.BytesIO(original_data)).convert("RGB")
        
        # 处理掩码
        mask_image = process_mask_from_base64(original_image, mask_image_base64)
        if mask_image is None:
            return jsonify({
                'success': False,
                'message': '无法处理掩码图像'
            }), 400
        
        # 计算外接矩形
        bbox = calculate_mask_bbox(mask_image)
        if bbox is None:
            return jsonify({
                'success': False,
                'message': '掩码为空，请先绘制掩码区域'
            }), 400
        
        x, y, width, height = bbox
        
        # 计算自动padding
        auto_padding = calculate_auto_padding(bbox, MODEL_INPUT_SIZE)
        
        return jsonify({
            'success': True,
            'bbox': {
                'x': x,
                'y': y,
                'width': width,
                'height': height
            },
            'auto_padding': auto_padding,
            'model_input_size': MODEL_INPUT_SIZE
        })
    except Exception as e:
        error_msg = f"计算掩码信息时出错: {str(e)}"
        print(error_msg)
        traceback.print_exc()
        return jsonify({
            'success': False,
            'message': error_msg
        }), 500

@app.route('/api/generate', methods=['POST'])
def generate():
    """生成缺陷图像API"""
    global pipe
    
    if pipe is None:
        return jsonify({
            'success': False,
            'message': '请先加载模型！'
        }), 400
    
    try:
        data = request.json
        
        # 获取参数
        original_image_base64 = data.get('original_image')
        mask_image_base64 = data.get('mask_image')
        prompt = data.get('prompt', 'defect of crack')
        negative_prompt = data.get('negative_prompt', '')
        num_images = int(data.get('num_images', 4))
        guidance_scale = float(data.get('guidance_scale', 7.0))
        num_inference_steps = int(data.get('num_inference_steps', 28))
        padding_mask_crop = data.get('padding_mask_crop')  # 可以是 None 或数字
        
        if not original_image_base64 or not mask_image_base64:
            return jsonify({
                'success': False,
                'message': '请提供原始图片和掩码'
            }), 400
        
        # 解码原始图像
        original_data = base64.b64decode(original_image_base64.split(',')[1])
        original_image = Image.open(io.BytesIO(original_data)).convert("RGB")
        
        # 处理掩码
        mask_image = process_mask_from_base64(original_image, mask_image_base64)
        if mask_image is None:
            return jsonify({
                'success': False,
                'message': '无法处理掩码图像'
            }), 400
        
        # 计算掩码外接矩形和裁剪区域信息（用于前端显示）
        bbox = calculate_mask_bbox(mask_image)
        crop_info = None
        if bbox and padding_mask_crop is not None:
            try:
                padding = int(padding_mask_crop)
                if padding > 0:
                    x, y, width, height = bbox
                    # 计算裁剪区域（带padding）
                    crop_x = max(0, x - padding)
                    crop_y = max(0, y - padding)
                    crop_width = min(original_image.width - crop_x, width + 2 * padding)
                    crop_height = min(original_image.height - crop_y, height + 2 * padding)
                    crop_info = {
                        'x': crop_x,
                        'y': crop_y,
                        'width': crop_width,
                        'height': crop_height
                    }
            except (ValueError, TypeError):
                pass
        
        # 检查显存是否足够（基于每批4张的显存需求）
        batch_size = 4  # 每批最多生成4张
        if device == "cuda":
            # 强制同步，确保所有操作完成
            torch.cuda.synchronize()
            # 先清理一次显存
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            gc.collect()
            
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            free = total - reserved
            print(f"生成前显存状态: 已分配 {allocated:.2f} GB, 已保留 {reserved:.2f} GB, 总计 {total:.2f} GB, 空闲 {free:.2f} GB")
            
            # 每批4张需要的显存（根据 padding_mask_crop 调整）
            base_memory_per_batch = 5.0  # 基础显存需求
            if padding_mask_crop is not None:
                try:
                    padding_mask_crop_int = int(padding_mask_crop)
                    if padding_mask_crop_int > 0:
                        base_memory_per_batch += 2.0  # padding_mask_crop 需要额外显存
                except (ValueError, TypeError):
                    pass
            
            # 检查单批是否能生成
            if free < base_memory_per_batch * 0.7:
                return jsonify({
                    'success': False,
                    'message': f'显存严重不足！需要至少 {base_memory_per_batch:.1f} GB 来生成一批图片，但只有 {free:.2f} GB 可用。建议：1) 重启应用释放显存 2) 使用更小的图片 3) 不使用 padding_mask_crop'
                }), 400
        
        print(f"开始分批生成，总共 {num_images} 张，每批 {batch_size} 张，参数: prompt={prompt}, guidance_scale={guidance_scale}")
        
        # 分批生成图像
        result_images = []
        total_batches = (num_images + batch_size - 1) // batch_size  # 向上取整
        
        for batch_idx in range(total_batches):
            # 计算当前批次要生成的数量
            current_batch_size = min(batch_size, num_images - len(result_images))
            print(f"正在生成第 {batch_idx + 1}/{total_batches} 批，本批生成 {current_batch_size} 张...")
            
            # 清理显存缓存
            if device == "cuda":
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                gc.collect()
            
            # 生成当前批次图像（使用torch.no_grad避免保留梯度）
            with torch.no_grad():
                generate_kwargs = {
                    'prompt': prompt,
                    'image': original_image,
                    'mask_image': mask_image,
                    'negative_prompt': negative_prompt if negative_prompt else None,
                    'num_images_per_prompt': current_batch_size,
                    'guidance_scale': guidance_scale,
                    'num_inference_steps': num_inference_steps,
                    'height': MODEL_INPUT_SIZE,  # 设置模型输入高度
                    'width': MODEL_INPUT_SIZE,   # 设置模型输入宽度
                }
                
                # 如果提供了 padding_mask_crop，添加到参数中
                if padding_mask_crop is not None:
                    try:
                        padding_mask_crop_int = int(padding_mask_crop)
                        if padding_mask_crop_int > 0:
                            generate_kwargs['padding_mask_crop'] = padding_mask_crop_int
                    except (ValueError, TypeError):
                        pass  # 忽略无效的值
                
                # 调用pipeline生成图像（会自动贴回原图）
                result = pipe(**generate_kwargs)
                
                # 获取生成结果
                if hasattr(result, 'images'):
                    batch_images = result.images
                else:
                    batch_images = result
                
                # 添加到总结果中
                result_images.extend(batch_images)
                
                # 清理当前批次的结果对象
                del result, batch_images
            
            # 每批生成后清理显存
            if device == "cuda":
                torch.cuda.empty_cache()
                gc.collect()
                if batch_idx < total_batches - 1:  # 不是最后一批，显示显存状态
                    allocated = torch.cuda.memory_allocated() / 1024**3
                    print(f"第 {batch_idx + 1} 批完成，已生成 {len(result_images)} 张，显存: {allocated:.2f} GB")
        
        # 清理临时变量和显存
        del original_image, mask_image
        if device == "cuda":
            torch.cuda.empty_cache()
        
        # 保存结果并转换为base64
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for i, img in enumerate(result_images):
            # 保存到文件
            filename = f"sd3_inpaint_{timestamp}_{i:02d}.png"
            filepath = os.path.join(app.config['OUTPUT_FOLDER'], filename)
            img.save(filepath)
            
            # 转换为base64用于前端显示
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            result_images[i] = f"data:image/png;base64,{img_base64}"
        
        # 最终清理显存
        if device == "cuda":
            torch.cuda.empty_cache()
            allocated = torch.cuda.memory_allocated() / 1024**3
            print(f"生成后显存: 已分配 {allocated:.2f} GB")
        
        return jsonify({
            'success': True,
            'message': f'成功生成 {len(result_images)} 张图片！',
            'images': result_images,
            'output_dir': app.config['OUTPUT_FOLDER'],
            'crop_info': crop_info,  # 裁剪区域信息（如果有）
            'bbox': bbox  # 掩码外接矩形信息（如果有）
        })
    
    except Exception as e:
        error_msg = f"生成过程中出错: {str(e)}"
        print(error_msg)
        traceback.print_exc()
        return jsonify({
            'success': False,
            'message': error_msg
        }), 500

@app.route('/api/check_model', methods=['GET'])
def check_model():
    """检查模型是否已加载"""
    return jsonify({
        'loaded': pipe is not None
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6008, debug=True)

