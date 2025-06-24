import os
import base64
from io import BytesIO
import json
import time
from flask import Flask, render_template, request, send_from_directory, jsonify
from PIL import Image
import requests
import replicate
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get API Token
REPLICATE_API_TOKEN = os.environ.get('REPLICATE_API_TOKEN')
if not REPLICATE_API_TOKEN:
    raise ValueError("REPLICATE_API_TOKEN is not set in the .env file")

# Configure paths
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'static/output'
AVATAR_PATH = 'Model.jpg'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER



@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    # Get uploaded files
    tshirt = request.files.get('tshirt')
    pants = request.files.get('pants')
    if not tshirt or not pants:
        return jsonify({'error': 'Both clothing images required'}), 400
    
    # Save uploaded files
    tshirt_path = os.path.join(app.config['UPLOAD_FOLDER'], 'tshirt.jpg')
    pants_path = os.path.join(app.config['UPLOAD_FOLDER'], 'pants.jpg')
    tshirt.save(tshirt_path)
    pants.save(pants_path)
    
    # Start a background task to process the images
    job_id = f"job_{int(time.time())}"
    try:
        # Generate descriptions for the avatar and clothing
        avatar_description = "a full-body anime-style illustration of a young woman with tied brown hair, neutral expression, front-facing pose"

        tshirt_description = describe_image(tshirt_path)
        pants_description = describe_image(pants_path)
        
        # Create combined prompt
        prompt = craft_prompt(avatar_description, tshirt_description, pants_description)
        
        # Run the image generation
        result = generate_dressed_avatar(prompt, job_id)
        
        return jsonify({
            'job_id': job_id,
            'status': 'processing',
            'prompt': prompt
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def describe_image(image_path):
    """Get a description of an image using Replicate's BLIP model"""
    with open(image_path, "rb") as file:
        # Use Replicate to get the image description
        image_base64 = base64.b64encode(file.read()).decode("utf-8")
        
        output = replicate.run(
            "salesforce/blip:2e1dddc8621f72155f24cf2e0adbde548458d3cab9f00c0139eea840d0ac4746",
            input={"image": f"data:image/jpeg;base64,{image_base64}"}
        )
        
        return output


def craft_prompt(avatar_desc, tshirt_desc, pants_desc):
    """Create a prompt that describes the avatar wearing the clothing"""
    # Clean up descriptions
    avatar_desc = avatar_desc.strip()
    tshirt_desc = tshirt_desc.strip().replace("a ", "").replace("an ", "")
    pants_desc = pants_desc.strip().replace("a ", "").replace("an ", "")
    
    # Create the final prompt
    prompt = (
    f"Keep the exact appearance, pose, face, and body of the person from the reference image. "
    f"The character is a full-body anime-style girl with brown hair in a tied bun, standing straight and barefoot. "
    f"Replace their outfit with: {tshirt_desc} and {pants_desc}. "
    f"The person should look identical in style and structure, only the clothes must be changed. "
    f"Only the clothing must change. Do not modify hair, body shape, facial features, or pose. "
    f"Do not alter facial features, hair, body proportions, or background. "
    f"Clean white background, full-body shot, sharp anime illustration, flat shading."
    f"Full body shot, clean studio background, high detail, realistic lighting."
)
    return prompt

#ideogram-ai/ideogram-v3-turbo
def generate_dressed_avatar(prompt, job_id):
    """Generate a dressed avatar image using Replicate with Imagen-4"""
    # Run the image generation model
    output_url = replicate.run(
        "ideogram-ai/ideogram-v3-turbo", #CHANGEEEEEEEEEEEE HERE
        input={
            "prompt": prompt,
            "aspect_ratio": "3:4",  # Corresponds to previous 768x1024 resolution
            "safety_filter_level": "block_medium_and_above"
        }
    )
    
    # Save the generated image
    if output_url:
        response = requests.get(output_url)
        
        if response.status_code == 200:
            # Imagen-4 returns a PNG, convert it to JPG to match existing frontend logic
            img = Image.open(BytesIO(response.content))
            result_path_jpg = os.path.join(app.config['OUTPUT_FOLDER'], f"result_{job_id}.jpg")
            img.convert('RGB').save(result_path_jpg, 'jpeg')
            return {'result_url': f"/static/output/result_{job_id}.jpg"}
        
    return {'error': 'Failed to generate image'}


@app.route('/result/<job_id>', methods=['GET'])
def get_result(job_id):
    """Check if the result is available for a job"""
    result_path = os.path.join(app.config['OUTPUT_FOLDER'], f"result_{job_id}.jpg")
    if os.path.exists(result_path):
        return jsonify({
            'status': 'completed',
            'result_url': f"/static/output/result_{job_id}.jpg"
        })
    else:
        return jsonify({'status': 'processing'})

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    # Ensure avatar image exists
    if not os.path.exists(AVATAR_PATH):
        print(f"Warning: Avatar image '{AVATAR_PATH}' not found!")
    
    # Start the Flask server
    app.run(debug=True)