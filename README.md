# 🧠 Shapeshifter: Sparse HSL Rectangular Encoder

Shapeshifter is a Streamlit application that showcases a novel image compression format designed for PNG files. This innovative format leverages HSL (Hue, Saturation, Lightness) color space and rectangular encoding to achieve significant file size reduction while maintaining lossless compression in the HSL domain.

# ✨ Key Features

- **Lossless Compression in HSL Space**: The compression stage is completely lossless, with any precision loss occurring only during the initial RGB to HSL conversion. If the input image is already optimized for HSL, the entire process is lossless.
- **Significant Size Reduction**: On a test image of approximately 2000x3000 pixels, Shapeshifter achieves a file size reduction of 30-40% compared to the original PNG.
- **Interactive Analysis**: The Streamlit app provides detailed insights into the compression process, including:
- Per-channel (H, S, L) unique value analysis
- Rectangular encoding details with bit optimization
- Comprehensive compression statistics
- Visual comparison of original, HSL-converted, and reconstructed images
- Difference heatmaps for error analysis
- **Reconstruction Verification**: Includes a reconstruction test to verify the accuracy of the compressed format against the original image.
- Particularly effective for rectangles-based images, noisy images, photorealism, and soon, flat images and icons.

# 🚀 How It Works

- **Image Upload**: Upload a PNG image via the Streamlit interface.
- **RGB to HSL Conversion**: The image is converted from RGB to HSL color space, with potential minor precision loss due to floating-point calculations.
- **HSL Quantization**: Hue (0-360), Saturation (0-100), and Lightness (0-100) are quantized to integers for efficient encoding.

- **Rectangular Encoding**: Each HSL channel is analyzed to identify rectangular regions of uniform values, which are encoded using:
- *A lookup table for unique values with optimized bit allocation*
- *Relative coordinates and differential size encoding for rectangles*
- *Bit-efficient encoding for zero values and inherited sizes*

- **Compression Analysis**: Detailed statistics on unique values, rectangle counts, bit usage, and compression savings are displayed.
- **Reconstruction Test**: The compressed data is reconstructed back into an image, with accuracy metrics and difference heatmaps provided for comparison.

# 🛠️ Installation
To run Shapeshifter locally, follow these steps:

***Clone the Repository***:
```
git clone https://github.com/SmilleCreeper/shapeshifter.git
cd shapeshifter
```

***Run the Streamlit App***:
```
streamlit run app.py
```


# 📋 Requirements

The requirements includes the following dependencies:
```
streamlit
numpy
Pillow
pandas
```

# 🎮 Usage

1. **Launch the app using streamlit run app.py.**
2. Upload a PNG image using the file uploader.
3. View the compression analysis, including unique value counts, bit usage, and rectangle details.
4. Use the "Reconstruct Image" button to verify the compression accuracy.
5. Filter the detailed rectangle analysis by channel, minimum area, or savings to explore the encoding process.

# 📊 Performance

On a test image of ~2000x3000 pixels, Shapeshifter achieves a 30-40% reduction in file size compared to the original PNG. The compression is most effective for images with large areas of uniform color in HSL space, as the rectangular encoding optimizes for these regions.

# ⚠️ Limitations

**RGB to HSL Conversion**: Minor precision loss may occur during the RGB to HSL conversion due to floating-point arithmetic. This can be mitigated by starting with HSL-optimized images.
**PNG-Specific**: The current implementation is optimized for PNG files and assumes lossless input.

# 🔮 Future Improvements

Support additional image formats (e.g., JPEG).
Implement parallel processing for HSL conversion and encoding.
Add export functionality for the compressed format.
Add color palette mappings to reduce size for flat images.
Break down Streamlit app into separate modules.
Button to process the file so it doesn't repeat every time the UI is updated.
