# Arisa AI Avatar

A web app that lets you upload your own clothes and see them fitted onto a base avatar using AI (FLUX.1-dev).

## Features
- Upload T-shirt and pants images
- See your clothes on a base avatar
- Powered by black-forest-labs/FLUX.1-dev (Stable Diffusion)

## Setup
0. Create a virtual environment:
   ```
   py -3.10 -m venv venv

   activate:

   .\venv\Scripts\activate
   ```

1. Install requirements:
   ```
   pip install -r requirements.txt
   ```
2. Run the app:
   ```
   py app.py
   ```
3. Open in browser: http://localhost:5000

## Folders
- `uploads/`: User-uploaded images
- `static/output/`: AI-generated results

