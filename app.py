from potassium import Potassium, Request, Response
import os
import cv2
import torch
import shutil
import base64
import tempfile
import numpy as np
from PIL import Image
from io import BytesIO
from gfpgan import GFPGANer
from realesrgan.utils import RealESRGANer
from basicsr.archs.srvgg_arch import SRVGGNetCompact

app = Potassium("my_app")

# @app.init runs at startup, and loads models into the app's context
@app.init
def init():
    device = 0 if torch.cuda.is_available() else -1
    # Load models
    model = SRVGGNetCompact(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_conv=32,
        upscale=4,
        act_type='prelu'
    )
    model_path = 'gfpgan/weights/realesr-general-x4v3.pth'
    half = True if torch.cuda.is_available() else False
    upsampler = RealESRGANer(
        scale=4,
        model_path=model_path,
        model=model,
        tile=0,
        tile_pad=10,
        pre_pad=0,
        half=half
    )
    # Use GFPGAN for face enhancement
    face_enhancer = GFPGANer(
        model_path='gfpgan/weights/GFPGANv1.4.pth',
        upscale=2,
        arch='clean',
        channel_multiplier=2,
        bg_upsampler=upsampler
    )

    context = {
        "model": model,
        "upsampler": upsampler,
        "face_enhancer": face_enhancer,
    }

    return context

# @app.handler runs for every call
@app.handler("/")
def handler(context: dict, request: Request) -> Response:
    img = request.json.get("img")
    version = request.json.get("version", "v1.4")
    scale = request.json.get("scale", 2)
    print(version, scale)
    current_version = 'v1.4'
    weight = 0.5

    # Model, upscaler and face_enhancer
    model = context.get("model")
    upsampler = context.get("upsampler")
    face_enhancer = context.get("face_enhancer")
    output = None
    try:
        #decode base64 of img
        img = img.encode('utf-8')
        img = BytesIO(base64.b64decode(img))
        img = Image.open(img)
        #cv2 read image bytes to numpy array
        img = np.array(img)

        if len(img.shape) == 3 and img.shape[2] == 4:
            img_mode = 'RGBA'
        elif len(img.shape) == 2:
            img_mode = None
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            img_mode = None
        print("---Shape check---")

        h, w = img.shape[0:2]
        if h < 300:
            img = cv2.resize(img, (w * 2, h * 2), interpolation=cv2.INTER_LANCZOS4)
        print("---Height check---")

        if current_version != version:
            if version == 'v1.2':
                face_enhancer = GFPGANer(
                    model_path='gfpgan/weights/GFPGANv1.2.pth',
                    upscale=2,
                    arch='clean',
                    channel_multiplier=2,
                    bg_upsampler=upsampler)
                current_version = 'v1.2'
            elif version == 'v1.3':
                face_enhancer = GFPGANer(
                    model_path='gfpgan/weights/GFPGANv1.3.pth',
                    upscale=2,
                    arch='clean',
                    channel_multiplier=2,
                    bg_upsampler=upsampler)
                current_version = 'v1.3'
            elif version == 'v1.4':
                face_enhancer = GFPGANer(
                    model_path='gfpgan/weights/GFPGANv1.4.pth',
                    upscale=2,
                    arch='clean',
                    channel_multiplier=2,
                    bg_upsampler=upsampler)
                current_version = 'v1.4'
            elif version == 'RestoreFormer':
                face_enhancer = GFPGANer(
                    model_path='gfpgan/weights/RestoreFormer.pth',
                    upscale=2,
                    arch='RestoreFormer',
                    channel_multiplier=2,
                    bg_upsampler=upsampler)
        print("---Loaded Version---")

        try:
            _, _, output = face_enhancer.enhance(img, has_aligned=False, only_center_face=False, paste_back=True, weight=weight)
            output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB) # convert BGR to RGB
        except RuntimeError as error:
            print('Error', error)
        print("---Loaded face_enhancer---")

        try:
            if scale != 2:
                interpolation = cv2.INTER_AREA if scale < 2 else cv2.INTER_LANCZOS4
                h, w = img.shape[0:2]
                output = cv2.resize(output, (int(w * scale / 2), int(h * scale / 2)), interpolation=interpolation)
                output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB) # convert BGR to RGB
        except Exception as error:
            print('wrong scale input.', error)
        print("---Try scale---")

        extension = 'jpg'
        if img_mode == 'RGBA':  # RGBA images should be saved in png format
            extension = 'png'
        save_path = 'out.' + extension
        cv2.imwrite(save_path, output)
        print("---Saved image---")
        with open(save_path, 'rb') as file:
            image_bytes = file.read()
        output = base64.b64encode(image_bytes)
        output = output.decode('utf-8')
        print("---Output base64---")
        # print(output)
        return Response(
            json = {"output": output}, 
            status=200
        )

    except Exception as error:
            print('global exception: ', error)

    return Response(
            json = {"output": output}, 
            status=200
        )

if __name__ == "__main__":
    app.serve()
