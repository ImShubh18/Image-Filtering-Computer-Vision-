# --- Existing Imports ---
from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
from PIL import Image, ImageFilter, ImageEnhance, ImageOps
import io
import numpy as np
from scipy import ndimage
from scipy.ndimage import gaussian_filter, median_filter, uniform_filter
from skimage import filters, segmentation, feature, measure
from skimage.color import rgb2gray
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from sklearn.cluster import KMeans
import cv2

# --- NEW: Imports for DB and File Handling ---
import os
import uuid
from database import Database, ImageLog, create_log_entry


# --- Flask App Setup ---
app = Flask(__name__)
CORS(app)

# --- NEW: Directory and Database Configuration ---
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# --- !! IMPORTANT !! ---
# PASTE YOUR MONGODB ATLAS CONNECTION STRING HERE
# mongodb+srv://shubham_adhav_dbuser:<db_password>@cluster01.qvp6a8g.mongodb.net/?retryWrites=true&w=majority&appName=Cluster01
MONGO_URI = "mongodb+srv://shubham_adhav_dbuser:dbuser1234@cluster01.qvp6a8g.mongodb.net/?retryWrites=true&w=majority"
DB_NAME = "pixel_codex_db"

# Initialize database connection
db = Database(MONGO_URI, DB_NAME)
if not db.client:
    # If connection fails, you might want to exit or handle it gracefully
    print("CRITICAL: Database connection failed. The app will run without logging.")


@app.route('/')
def home():
    return send_from_directory('.', 'index.html')

# ===== ORIGINAL FILTER FUNCTIONS (ALL LOGIC INCLUDED) =====

def apply_blur(image):
    """Apply blur filter to image"""
    return image.filter(ImageFilter.BLUR)

def apply_sharpen(image):
    """Apply sharpen filter to image"""
    return image.filter(ImageFilter.SHARPEN)

def apply_edge(image):
    """Apply edge detection filter to image"""
    return image.filter(ImageFilter.FIND_EDGES)

def apply_emboss(image):
    """Apply emboss filter to image"""
    return image.filter(ImageFilter.EMBOSS)

def apply_sepia(image):
    """Apply sepia filter to image"""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    pixels = np.array(image)
    sepia_filter = np.array([
        [0.393, 0.769, 0.189],
        [0.349, 0.686, 0.168],
        [0.272, 0.534, 0.131]
    ])
    
    sepia_pixels = pixels @ sepia_filter.T
    sepia_pixels = np.clip(sepia_pixels, 0, 255).astype(np.uint8)
    
    return Image.fromarray(sepia_pixels)

def apply_negative(image):
    """Apply negative filter to image"""
    return ImageOps.invert(image.convert('RGB'))

def apply_brightness(image):
    """Apply brightness enhancement to image"""
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(1.5)

def apply_contrast(image):
    """Apply contrast enhancement to image"""
    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(1.5)

# ===== NOISE REDUCTION FILTERS =====

def apply_median(image):
    """Apply median filter to image"""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    img_array = np.array(image)
    filtered = np.zeros_like(img_array)
    
    for i in range(3):  # For each color channel
        filtered[:,:,i] = median_filter(img_array[:,:,i], size=5)
    
    return Image.fromarray(filtered.astype(np.uint8))

def apply_gaussian(image):
    """Apply Gaussian filter to image"""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    img_array = np.array(image, dtype=np.float32)
    filtered = np.zeros_like(img_array)
    
    for i in range(3):
        filtered[:,:,i] = gaussian_filter(img_array[:,:,i], sigma=2.0)
    
    filtered = np.clip(filtered, 0, 255)
    return Image.fromarray(filtered.astype(np.uint8))

def apply_average(image):
    """Apply average/box filter to image"""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    img_array = np.array(image)
    filtered = np.zeros_like(img_array)
    
    for i in range(3):
        filtered[:,:,i] = uniform_filter(img_array[:,:,i], size=5)
    
    return Image.fromarray(filtered.astype(np.uint8))

# ===== ENHANCEMENT FILTERS =====

def apply_high_pass(image):
    """Apply high-pass filter to image"""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    img_array = np.array(image, dtype=np.float32)
    
    # High-pass kernel
    kernel = np.array([[-1, -1, -1],
                       [-1,  8, -1],
                       [-1, -1, -1]])
    
    filtered = np.zeros_like(img_array)
    for i in range(3):
        filtered[:,:,i] = ndimage.convolve(img_array[:,:,i], kernel)
    
    # Normalize and clip
    filtered = np.clip(filtered + 128, 0, 255)
    return Image.fromarray(filtered.astype(np.uint8))

def apply_high_boost(image):
    """Apply high-boost filter to image"""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    img_array = np.array(image, dtype=np.float32)
    
    # Apply Gaussian blur
    blurred = np.zeros_like(img_array)
    for i in range(3):
        blurred[:,:,i] = gaussian_filter(img_array[:,:,i], sigma=1.0)
    
    # High-boost: Original + A*(Original - Blurred)
    A = 1.5  # Amplification factor
    boosted = img_array + A * (img_array - blurred)
    boosted = np.clip(boosted, 0, 255)
    
    return Image.fromarray(boosted.astype(np.uint8))

def apply_sharpening(image):
    """Apply unsharp masking sharpening filter"""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    img_array = np.array(image, dtype=np.float32)
    
    # Unsharp mask kernel
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    
    filtered = np.zeros_like(img_array)
    for i in range(3):
        filtered[:,:,i] = ndimage.convolve(img_array[:,:,i], kernel)
    
    filtered = np.clip(filtered, 0, 255)
    return Image.fromarray(filtered.astype(np.uint8))

def apply_laplacian(image):
    """Apply Laplacian filter"""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    img_array = np.array(image, dtype=np.float32)
    
    # Laplacian kernel
    kernel = np.array([[0, -1, 0],
                       [-1, 4, -1],
                       [0, -1, 0]])
    
    filtered = np.zeros_like(img_array)
    for i in range(3):
        filtered[:,:,i] = ndimage.convolve(img_array[:,:,i], kernel)
    
    # Normalize and clip
    filtered = np.clip(filtered + 128, 0, 255)
    return Image.fromarray(filtered.astype(np.uint8))

def apply_sobel(image):
    """Apply Sobel edge detection"""
    if image.mode != 'RGB':
        image = image.convert('RGB')

    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel = np.sqrt(sobelx**2 + sobely**2)
    sobel = np.uint8(255 * sobel / np.max(sobel))
    sobel_rgb = np.stack([sobel, sobel, sobel], axis=2)
    return Image.fromarray(sobel_rgb)

def apply_prewitt(image):
    """Apply Prewitt edge detection"""
    if image.mode != 'RGB':
        image = image.convert('RGB')

    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    kernelx = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
    kernely = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    prewittx = cv2.filter2D(gray, -1, kernelx)
    prewitty = cv2.filter2D(gray, -1, kernely)
    prewitt = np.sqrt(prewittx**2 + prewitty**2)
    prewitt = np.uint8(255 * prewitt / np.max(prewitt))
    prewitt_rgb = np.stack([prewitt, prewitt, prewitt], axis=2)
    return Image.fromarray(prewitt_rgb)

# ===== EDGE DETECTION =====

def apply_canny(image):
    """Apply Canny edge detection"""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    img_array = np.array(image)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    
    # Convert to RGB
    edges_rgb = np.stack([edges, edges, edges], axis=2)
    return Image.fromarray(edges_rgb)

def apply_laplacian_edge(image):
    """Apply Laplacian of Gaussian edge detection"""
    if image.mode != 'RGB':
        image = image.convert('RGB')

    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
    laplacian = np.uint8(np.absolute(laplacian))
    laplacian_rgb = np.stack([laplacian, laplacian, laplacian], axis=2)
    return Image.fromarray(laplacian_rgb)

def apply_sobel_edge(image):
    """Apply Sobel edge detection (alias for consistency)"""
    return apply_sobel(image)

def apply_prewitt_edge(image):
    """Apply Prewitt edge detection (alias for consistency)"""
    return apply_prewitt(image)

# ===== SEGMENTATION =====

def apply_kmeans(image):
    """Apply K-means clustering for segmentation"""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    img_array = np.array(image)
    h, w, c = img_array.shape
    
    # Reshape image to be a list of pixels
    pixels = img_array.reshape((-1, 3))
    
    # Apply K-means clustering
    k = 8  # Number of clusters
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(pixels)
    
    # Replace each pixel with its cluster center
    segmented_pixels = kmeans.cluster_centers_[labels]
    segmented_image = segmented_pixels.reshape((h, w, c))
    
    return Image.fromarray(segmented_image.astype(np.uint8))

def apply_watershed(image):
    """Apply watershed segmentation"""
    if image.mode != 'RGB':
        image = image.convert('RGB')

    img_array = np.array(image)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

    # Noise removal
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Distance transform
    dist_transform = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(thresh, sure_fg)

    # Marker labelling
    num_markers, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    # Apply watershed
    markers = cv2.watershed(img_array, markers)
    img_array[markers == -1] = [255, 0, 0]  # Boundaries in red

    return Image.fromarray(img_array)

def apply_thresholding(image):
    """Apply adaptive thresholding"""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    img_array = np.array(image)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 11, 2)
    
    # Convert to RGB
    thresh_rgb = np.stack([thresh, thresh, thresh], axis=2)
    return Image.fromarray(thresh_rgb)

def apply_binary(image):
    """Convert image to binary using Otsu's thresholding"""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    img_array = np.array(image)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary_rgb = np.stack([binary, binary, binary], axis=2)
    return Image.fromarray(binary_rgb)

# ===== FEATURE DETECTION =====

def apply_orb(image):
    """Apply ORB feature detection"""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    img_array = np.array(image)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # Create ORB detector
    orb = cv2.ORB_create(nfeatures=500)
    keypoints = orb.detect(gray, None)
    
    # Draw keypoints
    result = cv2.drawKeypoints(img_array, keypoints, None, color=(0, 255, 0), flags=0)
    
    return Image.fromarray(result)

def apply_sift(image):
    """Apply SIFT feature detection"""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    img_array = np.array(image)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    try:
        # Create SIFT detector (requires opencv-contrib-python)
        sift = cv2.SIFT_create()
        keypoints = sift.detect(gray, None)
        
        # Draw keypoints
        result = cv2.drawKeypoints(img_array, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        return Image.fromarray(result)
    except AttributeError:
        # Fallback to Harris corners if SIFT not available
        corners = cv2.cornerHarris(gray, 2, 3, 0.04)
        result = img_array.copy()
        result[corners > 0.01 * corners.max()] = [0, 255, 0]
        return Image.fromarray(result)

def apply_surf(image):
    """Apply SURF feature detection (fallback to ORB if not available)"""
    # SURF is patented and not available in standard OpenCV
    # Using ORB as fallback
    return apply_orb(image)

# ===== TRANSFORMS =====

def apply_grayscale(image):
    """Convert image to grayscale"""
    gray = image.convert('L')
    return gray.convert('RGB')

def apply_brightening(image):
    """Apply brightening transformation"""
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(1.8)  # Increase brightness by 80%

def apply_darkening(image):
    """Apply darkening transformation"""
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(0.5)  # Decrease brightness by 50%

def apply_gray_level_slicing(image):
    """Apply gray level slicing"""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    img_array = np.array(image)
    gray = rgb2gray(img_array)
    
    # Define slicing range
    low_threshold = 0.3
    high_threshold = 0.7
    
    # Apply slicing
    sliced = np.zeros_like(gray)
    mask = (gray >= low_threshold) & (gray <= high_threshold)
    sliced[mask] = 1.0
    
    sliced_img = (sliced * 255).astype(np.uint8)
    
    # Convert to RGB
    sliced_rgb = np.stack([sliced_img, sliced_img, sliced_img], axis=2)
    return Image.fromarray(sliced_rgb)

def apply_negation(image):
    """Apply negation transformation"""
    return ImageOps.invert(image.convert('RGB'))


# --- Dictionary mapping filter names to functions ---
FILTER_FUNCTIONS = {
    'blur': apply_blur, 'sharpen': apply_sharpen, 'edge': apply_edge,
    'emboss': apply_emboss, 'sepia': apply_sepia, 'negative': apply_negative,
    'brightness': apply_brightness, 'contrast': apply_contrast, 'median': apply_median,
    'gaussian': apply_gaussian, 'average': apply_average, 'high_pass': apply_high_pass,
    'high_boost': apply_high_boost, 'sharpening': apply_sharpening,
    'laplacian': apply_laplacian, 'sobel': apply_sobel, 'prewitt': apply_prewitt,
    'canny': apply_canny, 'laplacian_edge': apply_laplacian_edge,
    'sobel_edge': apply_sobel_edge, 'prewitt_edge': apply_prewitt_edge,
    'kmeans': apply_kmeans, 'watershed': apply_watershed, 'thresholding': apply_thresholding,
    'binary': apply_binary, 'orb': apply_orb, 'sift': apply_sift, 'surf': apply_surf,
    'grayscale': apply_grayscale, 'brightening': apply_brightening,
    'darkening': apply_darkening, 'gray_level_slicing': apply_gray_level_slicing,
    'negation': apply_negation
}

# --- MODIFIED /apply_filter ENDPOINT ---
@app.route('/apply_filter', methods=['POST'])
def apply_filter():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        if 'filter' not in request.form:
            return jsonify({'error': 'No filter specified'}), 400
        
        image_file = request.files['image']
        filter_name = request.form['filter']
        
        if filter_name not in FILTER_FUNCTIONS:
            return jsonify({'error': f'Invalid filter: {filter_name}'}), 400
        
        try:
            image = Image.open(image_file.stream)
        except Exception as e:
            return jsonify({'error': f'Invalid image file: {str(e)}'}), 400

        # --- MongoDB Integration Logic Start ---
        
        unique_id = str(uuid.uuid4())
        file_ext = os.path.splitext(image_file.filename)[1] or '.jpg'
        original_filename = f"{unique_id}_original{file_ext}"
        processed_filename = f"{unique_id}_{filter_name}{file_ext}"

        original_path = os.path.join(UPLOAD_FOLDER, original_filename)
        processed_path = os.path.join(PROCESSED_FOLDER, processed_filename)

        save_image = image.copy()
        if save_image.mode in ("RGBA", "P"):
             save_image = save_image.convert("RGB")
        save_image.save(original_path)

        # --- Apply the filter ---
        filtered_image = FILTER_FUNCTIONS[filter_name](image)

        # --- Save filtered image and log to DB ---
        save_filtered_image = filtered_image.copy()
        if save_filtered_image.mode in ("RGBA", "P", "L"):
            save_filtered_image = save_filtered_image.convert("RGB")
        save_filtered_image.save(processed_path)

        if db.client:
            log_entry = ImageLog(
                original_filename=original_filename,
                processed_filename=processed_filename,
                filter_applied=filter_name
            )
            create_log_entry(db.logs_collection, log_entry)
        
        # --- MongoDB Integration Logic End ---

        # --- Send response back to the client ---
        img_io = io.BytesIO()
        try:
            format_to_save = filtered_image.format if filtered_image.format else 'JPEG'
            filtered_image.save(img_io, format_to_save, quality=90)
        except Exception:
            filtered_image.convert("RGB").save(img_io, 'JPEG', quality=90)
        
        img_io.seek(0)
        
        return send_file(
            img_io,
            mimetype=f'image/{format_to_save.lower()}',
            as_attachment=False,
            download_name=f'filtered_{filter_name}.jpg'
        )
        
    except Exception as e:
        print(f"Server error: {e}")
        return jsonify({'error': f'An unexpected server error occurred: {str(e)}'}), 500

# --- UNCHANGED ENDPOINTS ---
@app.route('/filters', methods=['GET'])
def get_available_filters():
    """Return list of available filters organized by category"""
    categories = {
        'basic': ['blur', 'sharpen', 'edge', 'emboss', 'sepia', 'negative', 'brightness', 'contrast'],
        'noise_reduction': ['median', 'gaussian', 'average'],
        'enhancement': ['high_pass', 'high_boost', 'sharpening', 'laplacian', 'sobel', 'prewitt'],
        'edge_detection': ['canny', 'laplacian_edge', 'sobel_edge', 'prewitt_edge'],
        'segmentation': ['kmeans', 'watershed', 'thresholding', 'binary'],
        'feature_detection': ['orb', 'sift', 'surf'],
        'transforms': ['grayscale', 'brightening', 'darkening', 'gray_level_slicing', 'negation']
    }
    
    return jsonify({
        'filters': list(FILTER_FUNCTIONS.keys()),
        'categories': categories,
        'total_filters': len(FILTER_FUNCTIONS)
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    print("Starting Enhanced Image Filter API with MongoDB Logging...")
    
    categories = {
        'Basic Filters': ['blur', 'sharpen', 'edge', 'emboss', 'sepia', 'negative', 'brightness', 'contrast'],
        'Noise Reduction': ['median', 'gaussian', 'average'],
        'Enhancement': ['high_pass', 'high_boost', 'sharpening', 'laplacian', 'sobel', 'prewitt'],
        'Edge Detection': ['canny', 'laplacian_edge', 'sobel_edge', 'prewitt_edge'],
        'Segmentation': ['kmeans', 'watershed', 'thresholding', 'binary'],
        'Feature Detection': ['orb', 'sift', 'surf'],
        'Transforms': ['grayscale', 'brightening', 'darkening', 'gray_level_slicing', 'negation']
    }
    
    for category, filters in categories.items():
        print(f"\n{category}:")
        print(f"  {', '.join(filters)}")
    
    print("\nServer running on http://127.0.0.1:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)