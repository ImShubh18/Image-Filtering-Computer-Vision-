Pixel Codex 🧪✨
(Formerly Image Filter Studio Pro)

An advanced web-based image processing tool that not only applies a wide range of filters but also provides a deep dive into the mathematical theory behind each operation. Perfect for students, developers, and anyone curious about the science of digital imaging.

🖼️ Live Demo & Screenshot
A live demonstration of Pixel Alchemist in action. The interface allows for easy image upload, filter selection, and a clear comparison between the original and processed images, with the mathematical theory readily available.

(Will be uploaded soon!!)

🚀 Key Features
📷 Easy Image Upload: Drag & drop or click to upload your images.

📚 Extensive Filter Library: Over 35 filters across 7 categories, from basic blurs to advanced feature detection.

📖 Interactive Theory Panel: Select a filter to instantly view the underlying mathematical formulas, convolution kernels, and algorithms, beautifully rendered with KaTeX.

🔍 Side-by-Side Comparison: Immediately see the effect of your chosen filter next to the original image.

💾 Download Your Work: Save the processed image directly from the browser.

📱 Responsive Design: A clean, modern UI that works on both desktop and mobile devices.

🛠️ Tech Stack
This project is built with a powerful combination of frontend and backend technologies:

Area	Technologies Used
Frontend	HTML5, CSS3, JavaScript (ES6), KaTeX
Backend	Python, Flask, Pillow (PIL), NumPy, SciPy, scikit-image, scikit-learn, OpenCV
Export to Sheets
⚙️ Setup and Installation
Follow these steps to get a local copy up and running.

Prerequisites
Python 3.9 or higher

pip (Python package installer)

Node.js (optional, for live-server)

Git

1. Clone the Repository
git clone https://github.com/your-username/pixel-alchemist.git
cd pixel-alchemist
2. Backend Setup
It's highly recommended to use a virtual environment.

# Create a virtual environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install the required Python packages
pip install -r requirements.txt
Note: If you don't have a requirements.txt file, create one in the root directory with the following content:

Flask
Flask-Cors
Pillow
numpy
scipy
scikit-image
scikit-learn
opencv-python
3. Frontend Setup
The frontend is composed of static files (index.html) and requires no special installation. However, you need to ensure the JavaScript fetch call points to the correct backend URL.

Open the index.html file.

Find the line const response = await fetch(...).

Make sure it points to your local Flask server: http://127.0.0.1:5000/apply_filter.

4. Running the Application
Start the Backend Server:

python your_backend_file.py
The Flask server will start on http://127.0.0.1:5000.

Start the Frontend: The easiest way is to use a live server. If you are using VS Code, you can use the Live Server extension.

Right-click on index.html.

Select "Open with Live Server". Your browser will open to an address like http://127.0.0.1:5500.

You can now use the application!

🕹️ How to Use
Upload: Drag an image onto the upload area or click to select a file.

Select a Filter: Expand a category in the "Filter Options" panel.

Learn: Click on a filter name. The "Mathematical Theory" panel will update with detailed information.

Apply: Click the "🚀 Apply Filter" button.

Compare & Download: The comparison panel will appear. You can view the result and click the "Download Processed Image" button to save it.

🔬 Available Filters
A comprehensive list of image processing operations available in the studio:

🎨 Basic Filters

Blur (Mean), Sharpen, Edge Detection, Emboss, Sepia, Negative, Brightness, Contrast

🛡️ Noise Reduction

Median Filter, Gaussian Blur, Average (Box)

🛠️ Enhancement

High-Pass, High-Boost, Unsharp Masking, Laplacian, Sobel, Prewitt

🎯 Edge Detection

Canny, Laplacian Edge, Sobel Edge, Prewitt Edge

🧩 Segmentation

K-Means, Watershed, Thresholding, Binary (Otsu)

✨ Feature Detection

ORB, SIFT, SURF

🔄 Transforms

Grayscale, Brightening, Darkening, Gray-Level Slicing, Negation

🤝 Contributing
Contributions, issues, and feature requests are welcome! Feel free to check the issues page.
