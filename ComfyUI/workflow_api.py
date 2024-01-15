import os
import random
import sys
from typing import Sequence, Mapping, Any, Union
import torch
import gradio as gr
from PIL import Image 

def get_value_at_index(obj: Union[Sequence, Mapping], index: int) -> Any:
    """Returns the value at the given index of a sequence or mapping.

    If the object is a sequence (like list or string), returns the value at the given index.
    If the object is a mapping (like a dictionary), returns the value at the index-th key.

    Some return a dictionary, in these cases, we look for the "results" key

    Args:
        obj (Union[Sequence, Mapping]): The object to retrieve the value from.
        index (int): The index of the value to retrieve.

    Returns:
        Any: The value at the given index.

    Raises:
        IndexError: If the index is out of bounds for the object and the object is not a mapping.
    """
    try:
        return obj[index]
    except KeyError:
        return obj["result"][index]


def find_path(name: str, path: str = None) -> str:
    """
    Recursively looks at parent folders starting from the given path until it finds the given name.
    Returns the path as a Path object if found, or None otherwise.
    """
    # If no path is given, use the current working directory
    if path is None:
        path = os.getcwd()

    # Check if the current directory contains the name
    if name in os.listdir(path):
        path_name = os.path.join(path, name)
        print(f"{name} found: {path_name}")
        return path_name

    # Get the parent directory
    parent_directory = os.path.dirname(path)

    # If the parent directory is the same as the current directory, we've reached the root and stop the search
    if parent_directory == path:
        return None

    # Recursively call the function with the parent directory
    return find_path(name, parent_directory)


def add_comfyui_directory_to_sys_path() -> None:
    """
    Add 'ComfyUI' to the sys.path
    """
    comfyui_path = find_path("ComfyUI")
    if comfyui_path is not None and os.path.isdir(comfyui_path):
        sys.path.append(comfyui_path)
        print(f"'{comfyui_path}' added to sys.path")


def add_extra_model_paths() -> None:
    """
    Parse the optional extra_model_paths.yaml file and add the parsed paths to the sys.path.
    """
    from main import load_extra_path_config

    extra_model_paths = find_path("extra_model_paths.yaml")

    if extra_model_paths is not None:
        load_extra_path_config(extra_model_paths)
    else:
        print("Could not find the extra_model_paths config file.")



def import_custom_nodes() -> None:
    """Find all custom nodes in the custom_nodes folder and add those node objects to NODE_CLASS_MAPPINGS

    This function sets up a new asyncio event loop, initializes the PromptServer,
    creates a PromptQueue, and initializes the custom nodes.
    """
    import asyncio
    import execution
    from nodes import init_custom_nodes
    import server

    # Creating a new event loop and setting it as the default loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Creating an instance of PromptServer with the loop
    server_instance = server.PromptServer(loop)
    execution.PromptQueue(server_instance)

    # Initializing custom nodes
    init_custom_nodes()


from nodes import (
    ControlNetApplyAdvanced,
    CLIPTextEncode,
    NODE_CLASS_MAPPINGS,
    ControlNetLoader,
    CheckpointLoaderSimple,
    LoadImage,
    KSampler,
    VAELoader,
    SaveImage,
    VAEDecode,
    EmptyLatentImage
)




def main(image, input_prompt, negative_prompt):
    add_comfyui_directory_to_sys_path()
    add_extra_model_paths()
    import_custom_nodes()
    print(image)
    input_image_path = LoadImage().save_input_from_gradio(image)
    print("input image will be stored at: " +input_image_path)
    output_image="output/"
    
    default_prompt = "portrait of an Asian, black hair, looking at viewer, plain background,  4k, uhd,(high detailed skin:1.1), 8k uhd, dslr, soft lighting, high quality, film grain, Fujifilm XT3, hairless, detailed shirt, relaxed face, detailed eyes, hairless"
    prompt_for_sd = input_prompt + default_prompt

    avoid_prompt = negative_prompt + "semi-realistic, (worst quality, low quality, normal quality:1.5), grayscale, monochrome, imperfect, blurry, low res, (semi-realistic, unrealistic, unreal, CGI, 3d render, airbrushed, pixelated, painting, cartoon, anime"


    with torch.inference_mode():
        checkpointloadersimple = CheckpointLoaderSimple()
        checkpointloadersimple_4 = checkpointloadersimple.load_checkpoint(
            ckpt_name="Model/photon_v1.safetensors"
        )

        emptylatentimage = EmptyLatentImage()
        emptylatentimage_5 = emptylatentimage.generate(
            width=image.width, height=image.height, batch_size=1
        )

        cliptextencode = CLIPTextEncode()
        cliptextencode_6 = cliptextencode.encode(
            text=prompt_for_sd,
            clip=get_value_at_index(checkpointloadersimple_4, 1),
        )

        cliptextencode_7 = cliptextencode.encode(
            text=avoid_prompt,
            clip=get_value_at_index(checkpointloadersimple_4, 1),
        )

        controlnetloader = ControlNetLoader()
        controlnetloader_11 = controlnetloader.load_controlnet(
            control_net_name="control_v11p_sd15_lineart_fp16.safetensors"
        )

        controlnetloader_15 = controlnetloader.load_controlnet(
            control_net_name="control_v11p_sd15_canny_fp16.safetensors"
        )

        loadimage = LoadImage()
        loadimage_21 = loadimage.load_image(image=input_image_path)

        ultralyticsdetectorprovider = NODE_CLASS_MAPPINGS[
            "UltralyticsDetectorProvider"
        ]()
        ultralyticsdetectorprovider_27 = ultralyticsdetectorprovider.doit(
            model_name="bbox/face_yolov8m.pt"
        )

        samloader = NODE_CLASS_MAPPINGS["SAMLoader"]()
        samloader_28 = samloader.load_model(
            model_name="sam_vit_b_01ec64.pth", device_mode="AUTO"
        )

        upscalemodelloader = NODE_CLASS_MAPPINGS["UpscaleModelLoader"]()
        upscalemodelloader_35 = upscalemodelloader.load_model(
            model_name="4x-UltraSharp.pth"
        )

        controlnetloader_55 = controlnetloader.load_controlnet(
            control_net_name="control_v11p_sd15_softedge_fp16.safetensors"
        )

        loadimage_62 = loadimage.load_image(image=input_image_path)

        vaeloader = VAELoader()
        vaeloader_64 = vaeloader.load_vae(
            vae_name="vae-ft-mse-840000-ema-pruned.safetensors"
        )

        facerestoremodelloader = NODE_CLASS_MAPPINGS["FaceRestoreModelLoader"]()
        facerestoremodelloader_66 = facerestoremodelloader.load_model(
            model_name="codeformer.pth"
        )

        pidinetpreprocessor = NODE_CLASS_MAPPINGS["PiDiNetPreprocessor"]()
        controlnetapplyadvanced = ControlNetApplyAdvanced()
        cannyedgepreprocessor = NODE_CLASS_MAPPINGS["CannyEdgePreprocessor"]()
        lineartpreprocessor = NODE_CLASS_MAPPINGS["LineArtPreprocessor"]()
        ksampler = KSampler()
        vaedecode = VAEDecode()
        imageupscalewithmodel = NODE_CLASS_MAPPINGS["ImageUpscaleWithModel"]()
        reactorfaceswap = NODE_CLASS_MAPPINGS["ReActorFaceSwap"]()
        facedetailer = NODE_CLASS_MAPPINGS["FaceDetailer"]()
        facerestorecfwithmodel = NODE_CLASS_MAPPINGS["FaceRestoreCFWithModel"]()
        saveimage = SaveImage()

        pidinetpreprocessor_56 = pidinetpreprocessor.execute(
            safe="enable", resolution=512, image=get_value_at_index(loadimage_21, 0)
        )

        controlnetapplyadvanced_58 = controlnetapplyadvanced.apply_controlnet(
            strength=0.72,
            start_percent=0,
            end_percent=1,
            positive=get_value_at_index(cliptextencode_6, 0),
            negative=get_value_at_index(cliptextencode_7, 0),
            control_net=get_value_at_index(controlnetloader_55, 0),
            image=get_value_at_index(pidinetpreprocessor_56, 0),
        )

        cannyedgepreprocessor_42 = cannyedgepreprocessor.execute(
            low_threshold=100,
            high_threshold=200,
            resolution=512,
            image=get_value_at_index(loadimage_21, 0),
        )

        controlnetapplyadvanced_18 = controlnetapplyadvanced.apply_controlnet(
            strength=0.75,
            start_percent=0,
            end_percent=1,
            positive=get_value_at_index(controlnetapplyadvanced_58, 0),
            negative=get_value_at_index(controlnetapplyadvanced_58, 1),
            control_net=get_value_at_index(controlnetloader_15, 0),
            image=get_value_at_index(cannyedgepreprocessor_42, 0),
        )

        lineartpreprocessor_43 = lineartpreprocessor.execute(
            resolution=512, image=get_value_at_index(loadimage_21, 0), coarse ="enable"
        )

        controlnetapplyadvanced_19 = controlnetapplyadvanced.apply_controlnet(
            strength=0.81,
            start_percent=0,
            end_percent=1,
            positive=get_value_at_index(controlnetapplyadvanced_18, 0),
            negative=get_value_at_index(controlnetapplyadvanced_18, 1),
            control_net=get_value_at_index(controlnetloader_11, 0),
            image=get_value_at_index(lineartpreprocessor_43, 0),
        )

        ksampler_3 = ksampler.sample(
            seed=random.randint(1, 2**64),
            steps=30,
            cfg=10,
            sampler_name="euler",
            scheduler="normal",
            denoise=1,
            model=get_value_at_index(checkpointloadersimple_4, 0),
            positive=get_value_at_index(controlnetapplyadvanced_19, 0),
            negative=get_value_at_index(controlnetapplyadvanced_19, 1),
            latent_image=get_value_at_index(emptylatentimage_5, 0),
        )

        vaedecode_8 = vaedecode.decode(
            samples=get_value_at_index(ksampler_3, 0),
            vae=get_value_at_index(vaeloader_64, 0),
        )

        imageupscalewithmodel_36 = imageupscalewithmodel.upscale(
            upscale_model=get_value_at_index(upscalemodelloader_35, 0),
            image=get_value_at_index(loadimage_62, 0),
        )

        reactorfaceswap_33 = reactorfaceswap.execute(
            enabled=True,
            swap_model="inswapper_128.onnx",
            facedetection="retinaface_resnet50",
            face_restore_model="none",
            detect_gender_source="no",
            detect_gender_input="no",
            source_faces_index="0",
            input_faces_index="0",
            console_log_level=1,
            source_image=get_value_at_index(imageupscalewithmodel_36, 0),
            input_image=get_value_at_index(vaedecode_8, 0),
        )

        facedetailer_24 = facedetailer.doit(
            guide_size=256,
            guide_size_for=True,
            max_size=768,
            seed=random.randint(1, 2**64),
            steps=20,
            cfg=8,
            sampler_name="euler",
            scheduler="normal",
            denoise=0.5,
            feather=5,
            noise_mask=True,
            force_inpaint=True,
            bbox_threshold=0.5,
            bbox_dilation=10,
            bbox_crop_factor=3,
            sam_detection_hint="center-1",
            sam_dilation=0,
            sam_threshold=0.93,
            sam_bbox_expansion=0,
            sam_mask_hint_threshold=0.87,
            sam_mask_hint_use_negative="False",
            drop_size=10,
            wildcard="",
            image=get_value_at_index(reactorfaceswap_33, 0),
            model=get_value_at_index(checkpointloadersimple_4, 0),
            clip=get_value_at_index(checkpointloadersimple_4, 1),
            vae=get_value_at_index(vaeloader_64, 0),
            positive=get_value_at_index(cliptextencode_6, 0),
            negative=get_value_at_index(cliptextencode_7, 0),
            bbox_detector=get_value_at_index(ultralyticsdetectorprovider_27, 0),
            sam_model_opt=get_value_at_index(samloader_28, 0),
        )

        facerestorecfwithmodel_65 = facerestorecfwithmodel.restore_face(
            facedetection="retinaface_resnet50",
            codeformer_fidelity=0.5,
            facerestore_model=get_value_at_index(facerestoremodelloader_66, 0),
            image=get_value_at_index(reactorfaceswap_33, 0),
        )

        saveimage_39 = saveimage.save_images(
            filename_prefix="ComfyUI",
            images=get_value_at_index(facerestorecfwithmodel_65, 0),
        )
        print(saveimage_39)
        print(saveimage_39['ui'])
        output_image= output_image + saveimage_39["ui"]["images"][0]["filename"]
    return Image.open(output_image)




demo = gr.Interface(
    fn=main,
    inputs=[gr.Image(label=("Input image"),type='pil'), gr.Textbox(label="Your prompt"), gr.Textbox(label="Thing you want to avoid in output (Optional):")],
    outputs=[gr.Image(label="Output")],
)

demo.launch(server_port=3002,share=1)