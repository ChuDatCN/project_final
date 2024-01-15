# project_final

I)  Tải thư viện cần thiết
1. Tải các model cần thiết vào thư mục ../ComfyUI/models https://drive.google.com/drive/folders/10ug4yxX_ad-Nig0mEbXLojyTMrCCG95p?usp=sharing
2. Trong ../ComfyUI/custom_nodes, tải các thư viện cần thiết github ( git clone )
C1:  qua ComfyUI-Manager
- git clone ComfyUI-Manager
- python3 custom_nodes/ComfyUI-Manager/scripts/colab-dependencies.py

C2: Tải từng thư viện:
- comfyui_controlnet_aux: https://github.com/Fannovel16/comfyui_controlnet_aux 
- ComfyUI-AnimateDiff-Evolved: https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved 
- ComfyUI-Impact-Pack:  https://github.com/ltdrdata/ComfyUI-Impact-Pack 
- comfyui-reactor-node: https://github.com/Gourieff/comfyui-reactor-node 
- facerestore_cf: https://github.com/mav-rik/facerestore_cf 

3. Tại thư mục ../ComfyUI/AI 
 - git clone https://github.com/comfyanonymous/ComfyUI


II) Cài đặt môi trường
1. cu121, xformers,...
- pip install https://download.pytorch.org/whl/cu118/xformers-0.0.22.post4%2Bcu118-cp310-cp310-manylinux2014_x86_64.whl -r requirements.txt 
- pip install onnxruntime-gpu color-matcher simpleeval
- pip3 install -U xformers --index-url https://download.pytorch.org/whl/cu121

2. Gradio: pip install gradio


III) Chạy file thực thi chương trình
- python3 ../ComfyUI/workflow_api.py



