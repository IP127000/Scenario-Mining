# Scenario-Mining
Video Scenario Mining powered by VLM

##server端

```bash
pip install transformers=4.57.3 fastapi
uvicorn vlm_api:app --host 0.0.0.0 --port 3000
```
##客户端
```bash
pip install opencv-python-headless==4.11.0.86 gradio==5.44.0 requests==2.32.4
python vlm_scene_extractor.py -c config.json --host 0.0.0.0 --port 80
```
