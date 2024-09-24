import torch
from diffusers import StableDiffusionPipeline
model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id)
pipe = pipe.to("cuda")  
def generate_image(prompt, image_path, num_inference_steps=50, guidance_scale=7.5, height=512, width=512):
    with torch.autocast("cuda"):
        image = pipe(
            prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale
        ).images[0]
    
    image.save(image_path) 
prompt = "A beautiful, serene lake surrounded by snow-capped mountains during sunset."
image_path = "serene_lake.png"
generate_image(prompt, image_path, num_inference_steps=75, guidance_scale=8.5, height=768, width=768)



